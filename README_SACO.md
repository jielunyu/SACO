# SACO — Causal State Intervention for CLIP-ReID

Source code of the **causal state intervention** (Stage-2 counterfactual
image–text alignment) built on top of [CLIP-ReID](https://github.com/syl1998/CLIP-ReID).
Everything else (data loaders, ViT-B/16 backbone, solvers, evaluation) follows the
original CLIP-ReID code base.

## Idea

Person ReID descriptions contain a *motion state* (walking / riding a bicycle)
that is irrelevant to identity but highly correlated with the source domain.
SACO treats the state as a causal factor and intervenes on it at Stage 2:

1. A **fixed state block** (CLIP token embeddings of `walking on foot` or
   `riding a bicycle`) is inserted into the prompt before the final period:

   ```
   [SOS] A photo of a <ID context ×4> person <STATE ×3> . [EOS] pad …
   ```

   The block occupies what used to be padding, so the context length (77) and
   the position of `[EOS]` are unchanged. It is **not learnable**: Stage 1 still
   only trains the identity context tokens.

2. At Stage 2 two frozen text-anchor sets are cached for every identity:
   the **real** text (native state, e.g. walking for Market1501) and the
   **counterfactual** text (opposite state, riding).

3. The total loss is

   ```
   L = w_id·L_id + w_tri·L_tri
     + w_i2t·CE(img·text_realᵀ, y)
     + w_cf ·CE(img·text_cfᵀ,   y)
   ```

   The counterfactual term forces the visual embedding to match the identity
   description *even when the stated motion state is wrong*, i.e. it removes the
   state direction from the retrieval descriptor.

## Key files

| File | Content |
|---|---|
| `model/make_model_clipreid.py` | `PromptLearner` with the fixed state block; `encode_text`, `cache_text_features`; training `forward` returns `logits_real, logits_cf` |
| `loss/make_loss.py` | adds the counterfactual I2T term (`i2tscore_cf`) |
| `processor/processor_clipreid_stage2.py` | caches the two anchor sets, unpacks the new model outputs |
| `config/defaults.py` | `MODEL.CAUSAL_STATE`, `NATIVE_STATE`, `I2T_CF_LOSS_WEIGHT`, `STATE_WALK_TEXT`, `STATE_RIDE_TEXT` |
| `configs/person/vit_clipreid_causal_market.yml` | Market1501 experiment (native state = walking, λ_cf = 1.0) |

`processor/processor_clipreid_stage1.py` is unchanged.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `MODEL.CAUSAL_STATE` | `False` | enable the state block + counterfactual loss |
| `MODEL.NATIVE_STATE` | `'walk'` | motion state of the source training set (`'walk'` / `'ride'`) |
| `MODEL.I2T_CF_LOSS_WEIGHT` | `1.0` | weight λ_cf of the counterfactual term |
| `MODEL.STATE_WALK_TEXT` | `'walking on foot'` | must tokenize to the same number of CLIP tokens as `STATE_RIDE_TEXT` |
| `MODEL.STATE_RIDE_TEXT` | `'riding a bicycle'` | |

With `MODEL.CAUSAL_STATE False` the code reproduces the original CLIP-ReID
behaviour exactly (the default in `config/defaults.py`).

## Run (Market1501)

```bash
# train (Stage 1: 120 epochs, Stage 2: 60 epochs)
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py \
    --config_file configs/person/vit_clipreid_causal_market.yml

# test the final checkpoint
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py \
    --config_file configs/person/vit_clipreid_causal_market.yml
```

Edit `DATASETS.ROOT_DIR` / `OUTPUT_DIR` / `TEST.WEIGHT` in the config for your
environment. For a baseline run, add `MODEL.CAUSAL_STATE False` to the command
line.

## Environment

Python 3.8, PyTorch 1.10, `yacs`, `timm`, `ftfy`, `regex`, `tqdm`
(see the original CLIP-ReID README). CLIP ViT-B/16 weights are downloaded
automatically by `model/clip/clip.py`.