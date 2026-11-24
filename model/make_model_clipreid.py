import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding, cfg, clip_model)
        self.text_encoder = TextEncoder(clip_model)

        # Triple semantic addressing (identity / domain / state)
        self.use_triple = bool(getattr(cfg.MODEL, 'TRIPLE_ADDRESSING', False))
        # hard geometric removal at inference is opt-in and off by default:
        # it removes structural priors along with the domain/state signal
        self.eval_purify = bool(getattr(cfg.MODEL, 'EVAL_PURIFY', False))
        self.probe_temp = float(getattr(cfg.MODEL, 'PROBE_TEMP', 0.05))

        # Frozen text anchors used by the Stage-2 image-text alignment.
        # They are filled once by ``cache_text_features`` (called from the
        # Stage-2 processor) and never optimised: this keeps the identity
        # anchors stationary while the image encoder adapts.
        self.register_buffer("text_features_real", torch.zeros(num_classes, self.in_planes_proj))
        self.register_buffer("text_features_cf", torch.zeros(num_classes, self.in_planes_proj))
        self._text_features_ready = False

    def encode_text(self, label, is_counterfactual=False):
        """Text features for ``label`` with the native (or opposite) state."""
        prompts = self.prompt_learner(label, is_counterfactual=is_counterfactual)
        return self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

    @torch.no_grad()
    def cache_text_features(self, chunk_size=256):
        """Pre-compute the class-wise text anchors for Stage-2 training.

        Two sets are cached: ``text_features_real`` uses the native motion
        state of the source domain, ``text_features_cf`` the opposite
        (counterfactual) state.  When the causal state block is disabled the
        two sets are identical and only the standard CLIP-ReID I2T loss is used.
        """
        device = self.classifier.weight.device
        labels = torch.arange(self.num_classes, device=device)
        was_training = self.training
        self.eval()
        with torch.cuda.amp.autocast(enabled=True):
            feats_real = torch.cat(
                [self.encode_text(labels[i:i + chunk_size], is_counterfactual=False)
                 for i in range(0, self.num_classes, chunk_size)], dim=0)
            if self.prompt_learner.use_state:
                feats_cf = torch.cat(
                    [self.encode_text(labels[i:i + chunk_size], is_counterfactual=True)
                     for i in range(0, self.num_classes, chunk_size)], dim=0)
            else:
                feats_cf = feats_real
        self.text_features_real = feats_real
        self.text_features_cf = feats_cf
        self._text_features_ready = True
        if was_training:
            self.train()

    def _triple_address(self, image_features, image_features_proj, is_riding=None):
        """Decompose the visual tokens into identity / domain / state parts.

        Text probes live in CLIP's joint 512-d space, which is exactly the space
        of ``image_features_proj`` (the projected patch tokens).  Attention is
        therefore computed between the normalised projected tokens and the
        normalised probes, and the resulting weights are also used to pool the
        768-d vision tokens so that all three components live in the same space
        as the CLS identity feature.
        """
        patch_proj = image_features_proj[:, 1:, :]
        patch_vis = image_features[:, 1:, :]

        patch_proj_n = nn.functional.normalize(patch_proj.float(), dim=-1)
        p_domain, p_state = self.prompt_learner.get_probes(is_riding)

        comps = []
        for probe in (p_domain, p_state):
            probe_n = nn.functional.normalize(probe.float(), dim=-1)          # [1, 512]
            probe_n = probe_n.expand(patch_proj_n.shape[0], -1).unsqueeze(-1)  # [B, 512, 1]
            attn = torch.softmax(
                torch.bmm(patch_proj_n, probe_n).squeeze(-1) / self.probe_temp,
                dim=1)                                                       # [B, N]
            attn = attn.to(patch_vis.dtype)
            v_vis = torch.bmm(attn.unsqueeze(1), patch_vis).squeeze(1)       # [B, 768]
            v_proj = torch.bmm(attn.unsqueeze(1), patch_proj).squeeze(1)     # [B, 512]
            comps.append((v_vis, v_proj))

        v_id = image_features[:, 0]                                          # [B, 768] CLS
        return (v_id,
                comps[0][0], comps[1][0],                                    # domain/state, 768-d
                comps[0][1], comps[1][1])                                    # domain/state, 512-d

    @staticmethod
    def _remove_component(v, d, eps=1e-6):
        """Project the direction ``d`` out of every row of ``v``."""
        v32, d32 = v.float(), d.float()
        coef = (v32 * d32).sum(1, keepdim=True) / (d32 * d32).sum(1, keepdim=True).clamp_min(eps)
        return (v32 - coef * d32).to(v.dtype)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None,
                is_counterfactual=False, is_riding=None):
        if get_text == True:
            return self.encode_text(label, is_counterfactual=is_counterfactual)

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            if not self._text_features_ready:
                # Safety net: the Stage-2 processor normally caches these once
                # before the epoch loop, so this only fires on direct usage.
                self.cache_text_features()
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)

            # Frozen text anchors -> image-to-text logits.  ``logits_cf`` is the
            # counterfactual branch (same image, opposite motion state).
            tf_real = self.text_features_real.to(img_feature_proj.dtype)
            logits_real = img_feature_proj @ tf_real.t()
            if self.prompt_learner.use_state:
                tf_cf = self.text_features_cf.to(img_feature_proj.dtype)
                logits_cf = img_feature_proj @ tf_cf.t()
            else:
                logits_cf = None

            # Triple addressing: (identity, domain, state) components used by
            # the orthogonal loss.  Only the ViT token layout is supported.
            orth = None
            if self.use_triple and self.model_name == 'ViT-B-16':
                orth = self._triple_address(image_features, image_features_proj, is_riding)

            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, logits_real, logits_cf, orth

        else:
            if self.use_triple and self.eval_purify and self.model_name == 'ViT-B-16':
                # Inference-time geometric removal of the domain and state
                # directions from the retrieval descriptor.
                _, v_domain, v_state, v_domain_proj, v_state_proj = self._triple_address(
                    image_features, image_features_proj, is_riding)
                img_feature = self._remove_component(img_feature, v_domain)
                img_feature = self._remove_component(img_feature, v_state)
                img_feature_proj = self._remove_component(img_feature_proj, v_domain_proj)
                img_feature_proj = self._remove_component(img_feature_proj, v_state_proj)
                feat = self.bottleneck(img_feature)
                feat_proj = self.bottleneck_proj(img_feature_proj)

            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        own = self.state_dict()
        skipped = []
        for i in param_dict:
            key = i.replace('module.', '')
            # buffers such as the cached text anchors are rebuilt at Stage-2
            # start, and probe buffers did not exist in older checkpoints
            if key not in own or own[key].shape != param_dict[i].shape:
                skipped.append(key)
                continue
            own[key].copy_(param_dict[i])
        if skipped:
            print('Skipped {} non-matching key(s): {}'.format(len(skipped), skipped[:8]))
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, cfg=None, clip_model=None):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])
        self.num_class = num_class
        self.n_ctx = n_ctx
        self.n_cls_ctx = n_cls_ctx
        self.context_length = tokenized_prompts.shape[1]

        # ------------------------------------------------------------------
        # Causal state dictionary (Stage-2 counterfactual intervention).
        # Two *fixed* CLIP token blocks describe the motion state of a person.
        # They are frozen embeddings, not learnable parameters, so the identity
        # context tokens stay the only trainable prompt part (Stage 1 unchanged).
        # ------------------------------------------------------------------
        if cfg is None:
            use_state, native_state = False, 'walk'
            walk_text, ride_text = 'walking on foot', 'riding a bicycle'
        else:
            use_state = bool(cfg.MODEL.CAUSAL_STATE)
            native_state = cfg.MODEL.NATIVE_STATE
            walk_text = cfg.MODEL.STATE_WALK_TEXT
            ride_text = cfg.MODEL.STATE_RIDE_TEXT

        if native_state not in ('walk', 'ride'):
            raise ValueError(
                "MODEL.NATIVE_STATE must be 'walk' or 'ride', got {!r}".format(native_state))

        self.use_state = use_state
        self.native_state = native_state
        self.state_texts = {'walk': walk_text, 'ride': ride_text}

        state_embs = {}
        for key in ('walk', 'ride'):
            ids = clip.tokenize(self.state_texts[key]).cuda()
            eot = int(ids.argmax(dim=-1)[0].item())
            with torch.no_grad():
                emb = token_embedding(ids).type(dtype)
            # keep only the words between [SOS] and [EOS]
            state_embs[key] = emb[0, 1:eot, :].unsqueeze(0).contiguous()

        n_walk = state_embs['walk'].shape[1]
        n_ride = state_embs['ride'].shape[1]
        if n_walk != n_ride:
            raise ValueError(
                "STATE_WALK_TEXT ({!r}) and STATE_RIDE_TEXT ({!r}) must tokenize to the "
                "same number of CLIP tokens, got {} and {}.".format(
                    walk_text, ride_text, n_walk, n_ride))
        self.n_state = n_walk

        # fixed (non-learnable) state tokens, shape [1, n_state, D]
        self.register_buffer("state_walk", state_embs['walk'])
        self.register_buffer("state_ride", state_embs['ride'])

        # ------------------------------------------------------------------
        # Fixed semantic probes for triple addressing: one domain probe and two
        # motion-state probes, encoded once by the frozen CLIP text encoder and
        # kept as buffers (they are anchors, never optimised).
        # ------------------------------------------------------------------
        self.use_triple = bool(cfg is not None and cfg.MODEL.TRIPLE_ADDRESSING)
        if cfg is not None:
            probe_texts = (cfg.MODEL.DOMAIN_PROBE_TEXT,
                           cfg.MODEL.STATE_WALK_PROBE_TEXT,
                           cfg.MODEL.STATE_RIDE_PROBE_TEXT)
        else:
            probe_texts = ('surveillance camera environment, background',
                           'a person walking',
                           'a person riding a bicycle')
        if clip_model is not None:
            with torch.no_grad():
                probes = [clip_model.encode_text(clip.tokenize(t).cuda()).type(dtype).detach().reshape(1, -1)
                          for t in probe_texts]
        else:
            probes = [torch.zeros(1, ctx_dim, dtype=dtype) for _ in probe_texts]
        self.register_buffer("domain_probe", probes[0])
        self.register_buffer("state_walk_probe", probes[1])
        self.register_buffer("state_ride_probe", probes[2])

        if use_state:
            # Prompt layout:
            #   [SOS] "A photo of a" | ID ctx (4) | "person" | STATE (n) | "." [EOS] pad ...
            # i.e. "A photo of a <id-context> person walking on foot." for the
            # native state, and "... person riding a bicycle." counterfactually.
            # The state block replaces what was pure padding, so the context
            # length (77) and the EOS position stay fixed.
            head_start = n_ctx + 1 + n_cls_ctx          # 9  -> "person"
            tail_start = head_start + 1                 # 10 -> "."
            tail_len = self.context_length - (n_ctx + 1 + n_cls_ctx + 1 + self.n_state)
            self.register_buffer("token_suffix_head", embedding[:, head_start:head_start + 1, :])
            self.register_buffer("token_suffix_tail", embedding[:, tail_start:tail_start + tail_len, :])

            # token ids with the state block inserted right before [EOS]; the
            # dummy ids are never embedded (only argmax -> EOS position is used)
            dummy = tokenized_prompts.new_full((1, self.n_state), 1)
            self.tokenized_prompts = torch.cat(
                [tokenized_prompts[:, :tail_start], dummy,
                 tokenized_prompts[:, tail_start:tail_start + tail_len]], dim=1)
            self.eot_idx = tail_start + self.n_state + 1  # tail[0] is "."
        else:
            self.eot_idx = int(tokenized_prompts.argmax(dim=-1)[0].item())

        assert self.tokenized_prompts.shape[1] == self.context_length

    def get_probes(self, is_riding=None):
        """Return the fixed (domain, state) probes for triple addressing.

        ``is_riding=None`` follows the source domain's native state, which is
        the right choice for single-domain training/inference.  Pass an explicit
        flag for joint walk/ride batches.
        """
        if not hasattr(self, 'domain_probe'):
            raise RuntimeError("PromptLearner was built without probes")
        if is_riding is None:
            state = self.state_walk_probe if self.native_state == 'walk' else self.state_ride_probe
        else:
            state = self.state_ride_probe if is_riding else self.state_walk_probe
        return self.domain_probe, state

    def forward(self, label, is_counterfactual=False):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)

        if not self.use_state:
            suffix = self.token_suffix.expand(b, -1, -1)
            return torch.cat([prefix, cls_ctx, suffix], dim=1)

        # real state = native state of the source domain; counterfactual = the
        # opposite state (Market/walking images -> "riding a bicycle" text).
        if is_counterfactual:
            state = self.state_ride if self.native_state == 'walk' else self.state_walk
        else:
            state = self.state_walk if self.native_state == 'walk' else self.state_ride
        state = state.expand(b, -1, -1)
        head = self.token_suffix_head.expand(b, -1, -1)
        tail = self.token_suffix_tail.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,      # [SOS] + "A photo of a"        (5, dim)
                cls_ctx,     # learnable identity context    (4, dim)
                head,        # "person" "."                  (2, dim)
                state,       # fixed motion-state words      (n, dim)
                tail,        # [EOS] + padding               (77-11-n, dim)
            ],
            dim=1,
        )

        return prompts 

