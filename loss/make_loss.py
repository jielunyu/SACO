# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def _cos2_orth_losses(orth):
        """Squared cosine between the identity feature and the domain/state
        components (all in the same 768-d vision space)."""
        v_id, v_domain, v_state = orth[0], orth[1], orth[2]

        def _cos2(a, b):
            a = F.normalize(a.float(), dim=1)
            b = F.normalize(b.float(), dim=1)
            return (a * b).sum(dim=1).pow(2).mean()

        return _cos2(v_id, v_domain), _cos2(v_id, v_state), _cos2(v_domain, v_state)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None, i2tscore_cf = None, orth = None):
            # ``i2tscore``     : image-text logits with the native state text
            # ``i2tscore_cf``  : image-text logits with the opposite (counterfactual)
            #                    state text; used by the causal intervention
            # ``orth``         : (v_id, v_domain, v_state) triple-addressing parts
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    I2TLOSS = None
                    I2T_CF_LOSS = None
                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    if i2tscore_cf is not None:
                        I2T_CF_LOSS = xent(i2tscore_cf, target)
                        loss = cfg.MODEL.I2T_CF_LOSS_WEIGHT * I2T_CF_LOSS + loss

                    ORTH_D = ORTH_S = ORTH_M = None
                    if orth is not None:
                        ORTH_D, ORTH_S, ORTH_M = _cos2_orth_losses(orth)
                        orth_total = ORTH_D + ORTH_S + cfg.MODEL.ORTH_MUTUAL_WEIGHT * ORTH_M
                        loss = cfg.MODEL.ORTH_LOSS_WEIGHT * orth_total + loss

                    loss_func.last_losses = {
                        'id': float(ID_LOSS.detach()),
                        'triplet': float(TRI_LOSS.detach()),
                        'i2t': None if I2TLOSS is None else float(I2TLOSS.detach()),
                        'i2t_cf': None if I2T_CF_LOSS is None else float(I2T_CF_LOSS.detach()),
                        'orth': None if ORTH_D is None else float(
                            (ORTH_D + ORTH_S + cfg.MODEL.ORTH_MUTUAL_WEIGHT * ORTH_M).detach()),
                        'orth_domain': None if ORTH_D is None else float(ORTH_D.detach()),
                        'orth_state': None if ORTH_S is None else float(ORTH_S.detach()),
                        'orth_mutual': None if ORTH_M is None else float(ORTH_M.detach()),
                    }

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    I2TLOSS = None
                    I2T_CF_LOSS = None
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    if i2tscore_cf is not None:
                        I2T_CF_LOSS = F.cross_entropy(i2tscore_cf, target)
                        loss = cfg.MODEL.I2T_CF_LOSS_WEIGHT * I2T_CF_LOSS + loss

                    ORTH_D = ORTH_S = ORTH_M = None
                    if orth is not None:
                        ORTH_D, ORTH_S, ORTH_M = _cos2_orth_losses(orth)
                        orth_total = ORTH_D + ORTH_S + cfg.MODEL.ORTH_MUTUAL_WEIGHT * ORTH_M
                        loss = cfg.MODEL.ORTH_LOSS_WEIGHT * orth_total + loss

                    loss_func.last_losses = {
                        'id': float(ID_LOSS.detach()),
                        'triplet': float(TRI_LOSS.detach()),
                        'i2t': None if I2TLOSS is None else float(I2TLOSS.detach()),
                        'i2t_cf': None if I2T_CF_LOSS is None else float(I2T_CF_LOSS.detach()),
                        'orth': None if ORTH_D is None else float(
                            (ORTH_D + ORTH_S + cfg.MODEL.ORTH_MUTUAL_WEIGHT * ORTH_M).detach()),
                        'orth_domain': None if ORTH_D is None else float(ORTH_D.detach()),
                        'orth_state': None if ORTH_S is None else float(ORTH_S.detach()),
                        'orth_mutual': None if ORTH_M is None else float(ORTH_M.detach()),
                    }

                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


