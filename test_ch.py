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
from tqdm import tqdm

from PIL import Image
import torchvision
def main(opt):
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)

    fix_seed(cfg.train.seed)
    logs = set_log(opt.output_dir, opt.cfg_file, opt.log_name)

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

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

    vae = AutoencoderKL.from_pretrained(opt.vae_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    model = ImitatingDiff(in_channels=cfg.model.in_channels, model_channels=cfg.model.emb_dims, 
                     out_channels=cfg.model.out_channels, num_res_blocks=cfg.model.num_res_blocks, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.model.num_heads, 
                     context_dim=cfg.model.emb_dims, style_encoder_layers= cfg.model.style_encoder_layers).to(device)

    if len(opt.pre_train) > 0:
        checkpoint = torch.load(opt.pre_train, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('load pretrained pre_train model from {}'.format(opt.pre_train))
        del checkpoint

    model = DDP(model, device_ids=[local_rank])

    model.eval()
    vae.eval()
    v_pbar = tqdm(test_loader, leave=False)
    tensor_to_pil = torchvision.transforms.ToPILImage()
    for step, test_data in enumerate(v_pbar):
        target_image, style_ref, mask_images, content_ref, style, target_image_name, origin_img_width, style_image_width = test_data['target_image'].to(device), \
            test_data['style_image'].to(device), \
            test_data['mask_image'].to(device), \
            test_data['content_image'].to(device), \
            test_data['style'], \
            test_data['target_image_name'], \
            test_data['origin_img_width'], \
            test_data['style_image_width']
        
        batch_size = content_ref.shape[0]
        x = torch.randn((batch_size, 16, content_ref.shape[2]//8, content_ref.shape[3]//8)).to(device)
        ema_sampled_images = diffusion.ddim_sample(model, vae, batch_size, x, style_ref, content_ref)

        out_path = logs['sample']
        os.makedirs(out_path, exist_ok=True)

        for index in range(len(ema_sampled_images)):
            s_img_width = style_image_width[index]
            o_img_width = origin_img_width[index]

            im = tensor_to_pil(ema_sampled_images[index])
            image = im.convert("L")
            style_path = os.path.join(out_path, style[index])
            os.makedirs(style_path, exist_ok=True)
            image = image.crop((0, 0, o_img_width, cfg.dataset.height))
            image.save(os.path.join(style_path, target_image_name[index] + "_predict.png"))

            t_im = target_image[index]
            t_im = (t_im / 2 + 0.5).clamp(0, 1)
            t_im = tensor_to_pil(t_im).convert("L")
            t_im = t_im.crop((0, 0, o_img_width, cfg.dataset.height))
            t_im.save(os.path.join(style_path, target_image_name[index] + "_target.png"))

            c_im = content_ref[index]
            c_im = (c_im / 2 + 0.5).clamp(0, 1)
            c_im = tensor_to_pil(c_im).convert("L")
            c_im = c_im.crop((0, 0, o_img_width, cfg.dataset.height))
            c_im.save(os.path.join(style_path, target_image_name[index] + "_content.png"))

            s_im = style_ref[index]
            s_im = (s_im / 2 + 0.5).clamp(0, 1)
            s_im = tensor_to_pil(s_im).convert("L")
            s_im = s_im.crop((0, 0, s_img_width, cfg.dataset.height))
            s_im.save(os.path.join(style_path, target_image_name[index] + "_style.png"))
            
        del test_data, ema_sampled_images
    v_pbar.close()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning if set")
    parser.add_argument('--vae_path', type=str, default='./.cache/sd3.5', help='path to stable diffusion 3.5 vae')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/Chinese_textline_vae16.yaml',
                        help='Config file for training and testing')
    parser.add_argument('--output', dest='output_dir', default='./output',
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
