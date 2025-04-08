import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
from engine_pretrain import boost_train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('BoostMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--bootstrap_stages', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch4', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--data_path', default='data/cifar10_data', type=str)
    parser.add_argument('--output_dir', default='./output_boost_dir')
    parser.add_argument('--log_dir', default='./output_boost_dir/tensorboard')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
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
            transforms.RandomCrop(args.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    try:
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    except Exception as e:
         print(f"Could not download/load CIFAR10 automatically from {args.data_path}. Error: {e}")
         try:
              print("Attempting to load dataset without download...")
              dataset_train = datasets.CIFAR10(args.data_path, train=True, download=False, transform=transform_train)
         except Exception as e2:
              print(f"Failed to load CIFAR10 from {args.data_path} even without download. Please check path and permissions. Error: {e2}")
              exit(1)

    print(dataset_train)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.bootstrap_stages <= 0:
        print("Error: bootstrap_stages must be positive.")
        exit(1)
    if args.epochs % args.bootstrap_stages != 0:
         print(f"Warning: Total epochs ({args.epochs}) not perfectly divisible by bootstrap_stages ({args.bootstrap_stages}).")
         print(f"Using floor division: {args.epochs // args.bootstrap_stages} epochs per stage.")
    epochs_per_stage = args.epochs // args.bootstrap_stages
    if epochs_per_stage == 0:
        print(f"Error: epochs_per_stage is zero ({args.epochs} // {args.bootstrap_stages}). Increase total epochs or decrease stages.")
        exit(1)
    print(f"Total epochs: {args.epochs}, Bootstrap stages: {args.bootstrap_stages}, Epochs per stage: {epochs_per_stage}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        print("Calculated absolute lr: %.2e" % args.lr)
    else:
         print("Using specified absolute lr: %.2e" % args.lr)

    print("Base lr provided (blr): %.2e" % args.blr)
    print("Effective batch size: %d" % eff_batch_size)
    print("Accumulate grad iterations: %d" % args.accum_iter)

    teacher_model_without_ddp = None
    final_model_state = None

    for stage in range(args.bootstrap_stages):
        print(f"\n{'='*20} Starting Bootstrap Stage {stage} {'='*20}")

        stage_output_dir = Path(args.output_dir) / f"stage_{stage}"
        stage_log_dir = Path(args.log_dir) / f"stage_{stage}"
        if global_rank == 0:
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            stage_log_dir.mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=str(stage_log_dir))
            print(f"Stage {stage} output dir: {stage_output_dir}")
            print(f"Stage {stage} log dir: {stage_log_dir}")
        else:
            log_writer = None

        current_stage_args = copy.deepcopy(args)
        current_stage_args.output_dir = str(stage_output_dir)
        current_stage_args.log_dir = str(stage_log_dir)
        current_stage_args.start_epoch = 0
        current_stage_args.epochs = epochs_per_stage
        if args.warmup_epochs >= epochs_per_stage:
             print(f"Warning: warmup_epochs ({args.warmup_epochs}) >= epochs_per_stage ({epochs_per_stage}). Setting stage warmup to {max(1, epochs_per_stage - 1)}.")
             stage_warmup_epochs = max(1, epochs_per_stage - 1)
        else:
             stage_warmup_epochs = args.warmup_epochs

        current_stage_args.warmup_epochs = max(0, stage_warmup_epochs)

        resume_path = None
        potential_resume_path = os.path.join(current_stage_args.output_dir, 'checkpoint.pth')
        if args.resume:
             if f"stage_{stage}" in args.resume and os.path.exists(args.resume):
                  resume_path = args.resume
                  print(f"Attempting to resume stage {stage} from specified checkpoint: {resume_path}")
             elif stage == 0 and "stage_" not in args.resume and os.path.exists(args.resume):
                 resume_path = args.resume
                 print(f"Attempting to resume stage 0 from specified general checkpoint: {resume_path}")

        if not resume_path and os.path.exists(potential_resume_path):
             resume_path = potential_resume_path
             print(f"Attempting to resume stage {stage} from default checkpoint: {resume_path}")

        if resume_path:
             current_stage_args.resume = resume_path

        use_norm_pix_loss_for_stage = args.norm_pix_loss and stage == 0
        student_model = models_mae.__dict__[args.model](
            norm_pix_loss=use_norm_pix_loss_for_stage
        )
        student_model.to(device)
        student_model_without_ddp = student_model
        print(f"Student Model (Stage {stage}) = %s" % str(student_model_without_ddp))
        print(f"Using norm_pix_loss: {use_norm_pix_loss_for_stage}")

        if stage > 0 and teacher_model_without_ddp is not None:
            print(f"Initializing student encoder from teacher (stage {stage-1})")
            teacher_state_dict = teacher_model_without_ddp.state_dict()
            student_state_dict = student_model_without_ddp.state_dict()

            encoder_keys = [k for k in teacher_state_dict if k in student_state_dict and \
                            (k.startswith('patch_embed.') or \
                             k.startswith('blocks.') or \
                             k.startswith('norm.') or \
                             k == 'cls_token' or \
                             k == 'pos_embed')]

            if not encoder_keys:
                 print("Warning: No matching encoder keys found between teacher and student.")
            else:
                print(f"Found {len(encoder_keys)} matching keys for encoder initialization.")

            load_state_dict = {k: teacher_state_dict[k] for k in encoder_keys}

            msg = student_model_without_ddp.load_state_dict(load_state_dict, strict=False)
            print(f"Encoder loading message: {msg}")

        if args.distributed:
            student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu], find_unused_parameters=True)
            student_model_without_ddp = student_model.module

        param_groups = optim_factory.add_weight_decay(student_model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=current_stage_args.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        print(f"Stage {stage} Optimizer: {optimizer}")
        print(f"Stage {stage} using Actual lr: %.2e" % current_stage_args.lr)
        print(f"Stage {stage} Warmup epochs: {current_stage_args.warmup_epochs}")
        print(f"Stage {stage} Total epochs in stage: {current_stage_args.epochs}")

        misc.load_model(args=current_stage_args, model_without_ddp=student_model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        current_teacher_model = None
        if stage > 0 and teacher_model_without_ddp is not None:
            current_teacher_model = teacher_model_without_ddp
            current_teacher_model.eval()
            print(f"Using teacher model from stage {stage-1} in eval mode.")

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        epochs_to_run = current_stage_args.epochs - current_stage_args.start_epoch
        if epochs_to_run <= 0:
             print(f"Stage {stage} already completed based on resume epoch {current_stage_args.start_epoch}. Skipping training.")
        else:
             print(f"Start training stage {stage} from epoch {current_stage_args.start_epoch} for {epochs_to_run} epochs")

        start_time = time.time()
        for epoch_in_stage in range(current_stage_args.start_epoch, current_stage_args.epochs):
            if args.distributed:
                global_epoch = stage * epochs_per_stage + epoch_in_stage
                data_loader_train.sampler.set_epoch(global_epoch)
            else:
                global_epoch = stage * epochs_per_stage + epoch_in_stage

            train_stats = boost_train_one_epoch(
                student_model, data_loader_train,
                optimizer, device, epoch_in_stage, loss_scaler,
                log_writer=log_writer,
                args=current_stage_args,
                teacher_model=current_teacher_model
            )

            if current_stage_args.output_dir and ((epoch_in_stage + 1) % 20 == 0 or epoch_in_stage + 1 == current_stage_args.epochs):
                 misc.save_model(
                    args=current_stage_args, model=student_model, model_without_ddp=student_model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch_in_stage)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'stage': stage,
                            'epoch_in_stage': epoch_in_stage,
                            'global_epoch': global_epoch}

            if current_stage_args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.add_scalar('misc/epoch_time', time.time() - start_time, log_stats['global_epoch'])
                    for k, v in train_stats.items():
                        log_writer.add_scalar(f'train/{k}', v, log_stats['global_epoch'])
                    log_writer.flush()

                with open(os.path.join(current_stage_args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            start_time = time.time()

        if log_writer is not None:
            log_writer.close()

        teacher_model_without_ddp = student_model_without_ddp

        if stage == args.bootstrap_stages - 1:
             print("Storing final model state.")
             final_model_state = copy.deepcopy(teacher_model_without_ddp.state_dict())

        print(f"Finished Bootstrap Stage {stage}")


    if args.output_dir and misc.is_main_process() and final_model_state is not None:
        final_model_path = os.path.join(args.output_dir, f'final_boostmae_model_stages_{args.bootstrap_stages}.pth')
        print(f"\nSaving final BoostMAE model state to {final_model_path}")
        save_obj = {
            'model': final_model_state,
            'args': args,
        }
        torch.save(save_obj, final_model_path)
        print("Final model saved.")
    elif misc.is_main_process():
         print("\nWarning: Final model state not saved (output_dir not set, not main process, or training didn't complete).")

    print("BoostMAE training process finished.")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)