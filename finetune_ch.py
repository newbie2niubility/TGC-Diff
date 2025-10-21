import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from utils.util import fix_seed
from utils.logger import set_log
from dataset.textline_dataset import TextLineDataset
from dataset.collate_fn import CollateFN
from torch.utils.data.distributed import DistributedSampler
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
from models.unet import ImitatingDiff
from trainer.trainer import Trainer
from torch import optim

def main(opt):
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)

    fix_seed(cfg.train.seed)
    logs = set_log(opt.output_dir, opt.cfg_file, opt.log_name)

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

    train_textline_dataset = TextLineDataset(
        args=cfg.dataset,
        mode='chinese',
        phase='train', 
    )
    train_sampler = DistributedSampler(train_textline_dataset)
    train_loader = torch.utils.data.DataLoader(train_textline_dataset,
                                               batch_size=cfg.train.imgs_per_batch,
                                               drop_last=False,
                                               collate_fn=CollateFN(),
                                               num_workers=cfg.dataloader.num_threads,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_textline_dataset = TextLineDataset(
        args=cfg.dataset,
        mode='chinese',
        phase='test', 
    )
    test_sampler = DistributedSampler(test_textline_dataset,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_textline_dataset,
                                               batch_size=cfg.test.imgs_per_batch,
                                               drop_last=False,
                                               collate_fn=CollateFN(),
                                               num_workers=cfg.dataloader.num_threads,
                                               pin_memory=True,
                                               sampler=test_sampler)

    valid_sampler = DistributedSampler(test_textline_dataset)
    valid_loader = torch.utils.data.DataLoader(test_textline_dataset,
                                               batch_size=cfg.test.imgs_per_batch,
                                               drop_last=False,
                                               collate_fn=CollateFN(),
                                               num_workers=cfg.dataloader.num_threads,
                                               pin_memory=True,
                                               sampler=valid_sampler)

    vae = AutoencoderKL.from_pretrained(opt.vae_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    model = ImitatingDiff(in_channels=cfg.model.in_channels, model_channels=cfg.model.emb_dims, 
                     out_channels=cfg.model.out_channels, num_res_blocks=cfg.model.num_res_blocks, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.model.num_heads, 
                     context_dim=cfg.model.emb_dims, style_encoder_layers= cfg.model.style_encoder_layers).to(device)

    # if len(opt.feat_model) > 0:
    #     checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'), weights_only=True)
    #     miss, unexp = model.style_encoder.load_state_dict(checkpoint, strict=False)
    #     print('load pretrained Style_Encoder from {}'.format(opt.feat_model))
    #     del checkpoint

    optimizer = optim.AdamW(model.parameters(), lr=cfg.solver.base_lr)
    if len(opt.pre_train) > 0:
        checkpoint = torch.load(opt.pre_train, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('load pretrained pre_train model from {}'.format(opt.pre_train))
        del checkpoint

    model = DDP(model, device_ids=[local_rank])

    trainer = Trainer(cfg, diffusion, model, vae, optimizer, train_loader, logs, valid_loader, device, opt.fine_tune)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning if set")
    parser.add_argument('--vae_path', type=str, default='./.cache/sd3.5', help='path to stable diffusion 3.5 vae')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/Chinese_textline_vae16.yaml',
                        help='Config file for training and testing')
    parser.add_argument('--output', dest='output_dir', default='/mnt/ShareDB-3TB/whl/hl/icdar2025_rep',
                        help='save logs and checkpoints')
    parser.add_argument('--feat_model', dest='feat_model', default='', help='pre-trained resnet18 model')
    parser.add_argument('--pre_train', dest='pre_train', default='', help='pre-trained model')
    parser.add_argument('--log_name', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)
