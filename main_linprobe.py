import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import models_vit
from engine_finetune import train_one_epoch, evaluate
import torch.nn as nn


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for CIFAR-10 classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--finetune', default='', required=True)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--data_path', default='./data/cifar10_data', type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--output_dir', default='./output_linprobe_dir')
    parser.add_argument('--log_dir', default='./output_linprobe_dir/tensorboard')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    try:
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    except Exception as e:
         print(f"Could not download/load CIFAR10 automatically from {args.data_path}. Error: {e}")
         print("Please ensure the dataset exists or path is correct.")
         exit(1)

    print(f"Dataset Train: {dataset_train}")
    print(f"Dataset Val: {dataset_val}")
    print(f"Number of classes: {args.nb_classes}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
             sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=args.global_pool,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'state_dict' in checkpoint:
             checkpoint_model = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
             checkpoint_model = checkpoint
        else:
             raise ValueError(f"Could not find model state_dict in checkpoint: {args.finetune}")

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
                del checkpoint_model[k]

        try:
            interpolate_pos_embed(model, checkpoint_model)
        except Exception as e:
            print(f"Warning: Position embedding interpolation failed: {e}. Check model compatibility.")
            print("Attempting to load without pos_embed if shapes mismatch...")
            pass

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        expected_missing = {'head.weight', 'head.bias'}
        if hasattr(model, 'fc_norm') and args.global_pool:
             expected_missing.update({'fc_norm.weight', 'fc_norm.bias'})

        if 'pos_embed' not in checkpoint_model or checkpoint_model['pos_embed'].shape != state_dict['pos_embed'].shape:
             if 'pos_embed' in msg.missing_keys:
                  print("Note: 'pos_embed' was missing or shape mismatched, loading without it.")
                  pass

        unexpected_missing = set(msg.missing_keys) - expected_missing
        if len(unexpected_missing) > 0:
             print(f"Error: Unexpected missing keys: {unexpected_missing}")


        trunc_normal_(model.head.weight, std=.02)
        nn.init.constant_(model.head.bias, 0)


    for name, p in model.named_parameters():
         if name.startswith('head'):
              p.requires_grad = True
         else:
              p.requires_grad = False

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('Number of trainable params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("Actual lr: %.2e" % args.lr)

    print("Accumulate grad iterations: %d" % args.accum_iter)
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module


    parameters_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
            parameters_to_optimize,
            lr=args.lr,
            betas=args.opt_betas if args.opt_betas is not None else (0.9, 0.999), 
            eps=args.opt_eps,
            weight_decay=args.weight_decay
        )

    print(f"Optimizer: {optimizer}")
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("Criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        print("Running evaluation only...")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if misc.is_main_process():
             if log_writer is not None:
                  log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], 0)
                  log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], 0)
                  log_writer.add_scalar('perf/test_loss', test_stats['loss'], 0)
             log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
             if args.output_dir:
                  with open(os.path.join(args.output_dir, "eval_log.txt"), mode="a", encoding="utf-8") as f:
                       f.write(json.dumps(log_stats) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
             misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)