import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    # -------------------------------------------------
    # device 设置
    # -------------------------------------------------
    device = "cuda"

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # -------------------------------------------------
    # Stage1: 先提取所有 image features
    # -------------------------------------------------
    import time
    from datetime import timedelta

    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    image_features = []
    labels = []

    model.eval()

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)

            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)

            for label_i, img_feat in zip(target, image_feature):
                labels.append(label_i.detach().cpu())
                image_features.append(img_feat.detach().cpu())

    if len(labels) == 0 or len(image_features) == 0:
        raise RuntimeError(
            "Stage1 error: no image features were extracted. "
            "Please check train_loader_stage1 and dataset parsing."
        )

    labels_list = torch.stack(labels, dim=0).cuda()              # [N]
    image_features_list = torch.stack(image_features, dim=0).cuda()  # [N, C]

    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]

    if num_image <= 0:
        raise RuntimeError(
            "Stage1 error: num_image <= 0. "
            "Please check whether the training dataset is empty."
        )

    if batch <= 0:
        raise RuntimeError(
            "Stage1 error: cfg.SOLVER.STAGE1.IMS_PER_BATCH must be > 0."
        )

    # 正确计算 batch 数，避免 num_image 能被 batch 整除时多跑一个空 batch
    num_batches = (num_image + batch - 1) // batch

    logger.info(
        "Stage1 feature cache finished: num_image={}, batch={}, num_batches={}".format(
            num_image, batch, num_batches
        )
    )

    del labels, image_features

    # -------------------------------------------------
    # Stage1: 用缓存的 image features 训练 text encoder
    # -------------------------------------------------
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)

        for i in range(num_batches):
            optimizer.zero_grad()

            start = i * batch
            end = min((i + 1) * batch, num_image)

            b_list = iter_list[start:end]

            # 理论上不会再出现空 batch，但保留保护
            if b_list.numel() == 0:
                logger.warning(
                    "Skip empty batch in Stage1: epoch={}, iter={}, start={}, end={}, num_image={}".format(
                        epoch, i, start, end, num_image
                    )
                )
                continue

            target = labels_list[b_list]
            image_features = image_features_list[b_list]

            if target.size(0) == 0 or image_features.size(0) == 0:
                logger.warning(
                    "Skip invalid batch in Stage1: target={}, image_features={}".format(
                        target.shape, image_features.shape
                    )
                )
                continue

            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True)

                if text_features.size(0) == 0:
                    logger.warning(
                        "Skip empty text_features in Stage1: epoch={}, iter={}".format(
                            epoch, i
                        )
                    )
                    continue

                loss_i2t = xent(image_features, text_features, target, target)
                loss_t2i = xent(text_features, image_features, target, target)

                loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), target.size(0))

            torch.cuda.synchronize()

            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        i + 1,
                        num_batches,
                        loss_meter.avg,
                        scheduler._get_lr(epoch)[0]
                    )
                )

        # -------------------------------------------------
        # 保存 checkpoint
        # -------------------------------------------------
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            cfg.OUTPUT_DIR,
                            cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)
                        )
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR,
                        cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)
                    )
                )

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))