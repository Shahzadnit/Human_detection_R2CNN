# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from PIL import Image


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # print(all_losses)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        # print(all_losses)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    # print(meters)
    max_iter = len(data_loader)
    # print(max_iter)
    start_iter = arguments["iteration"]
    # print(start_iter)
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, mask,idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        # print(targets[0].quad_bbox.device, targets[0].bbox.device)
        # print(targets)

        #mask=mask[0].to(device)
        mask=mask.to(device)
        #print(mask.tensors.shape)
        mask=(mask.tensors).long()
        # mask=(mask.tensors)
        images = images.to(device)
        #print(targets)
        targets = [target.to(device) for target in targets]
        #print(len(targets))
        # print(targets[0].bbox.shape)#torch.Size([23, 4])
        loss_dict = model(images, targets, mask)
        
        del mask, images, targets
        losses = sum(loss for loss in loss_dict.values())
        # print(losses.item())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print(**loss_dict_reduced)

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        # print(meters.get_avg())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        #scheduler.step(meters.get_avg())
        scheduler.step()
        # del losses
        

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


        # loss_list=[]
        # loss_list.append(losses.item())
        # thr=0.001
        # if iteration % 100==0:
        #     change=abs(loss_list[0]-loss_list[len(loss_list)-1])
        #     if change < thr:
        #         optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]/10
        #         # for param_group in optimizer.param_groups:
        #         #     param_group[0]['lr'] = optimizer.param_groups[0]["lr"]/10
        #     print("learning rate update-------->",optimizer.param_groups[0]["lr"])

        #     loss_list.clear()




        # print(optimizer.param_groups[0]["lr"])

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
        del loss_dict,loss_dict_reduced
        torch.cuda.empty_cache()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
