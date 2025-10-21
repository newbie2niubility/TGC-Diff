import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os
import sys
from PIL import Image
import torchvision
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

class Trainer:
    def __init__(self, cfg, diffusion, model, vae, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, fine_tune=False):
        self.cfg = cfg 
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.device = device
        self.fine_tune = fine_tune

    def _train_iter(self, data, step, pbar):
        self.model.train()
        target_image, style_ref, mask_images, content_ref = data['target_image'].to(self.device), \
            data['style_image'].to(self.device), \
            data['mask_image'].to(self.device), \
            data['content_image'].to(self.device)

        images = self.vae.encode(target_image).latent_dist.sample()
        images = images * 0.18215

        vae_content = self.vae.encode(content_ref).latent_dist.sample()
        vae_content = vae_content * 0.18215

        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)

        predicted_noise = self.model(x_t,  timesteps=t, style=style_ref, content=vae_content)

        latent_loss = F.mse_loss(predicted_noise.float(), noise.float(), reduction='none')
        latent_loss = torch.mean(latent_loss)

        # backward and update trainable parameters
        loss = latent_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if dist.get_rank() == 0:
            # log file
            loss_dict = {"latent_loss": latent_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        target_image, style_ref, mask_images, content_ref = data['target_image'].to(self.device), \
            data['style_image'].to(self.device), \
            data['mask_image'].to(self.device), \
            data['content_image'].to(self.device)

        images = self.vae.encode(target_image).latent_dist.sample()
        images = images * 0.18215

        vae_content = self.vae.encode(content_ref).latent_dist.sample()
        vae_content = vae_content * 0.18215

        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)

        predicted_noise = self.model(x_t,  timesteps=t, style=style_ref, content=vae_content)

        latent_loss = F.mse_loss(predicted_noise.float(), noise.float(), reduction='none')
        latent_loss = torch.mean(latent_loss)
    
        predicted_noise = 1 / 0.18215 * predicted_noise
        predicted_noise = self.vae.decode(predicted_noise).sample

        noise = 1 / 0.18215 * noise
        noise = self.vae.decode(noise).sample

        diff_loss = F.l1_loss(predicted_noise.float(), noise.float(), reduction='none')
        diff_loss = mask_images * diff_loss
        diff_loss = torch.mean(diff_loss)

        loss = latent_loss + 0.5 * diff_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"latent_loss": latent_loss.item(), "diff_loss": diff_loss.item(),  "total_loss": loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _valid_iter(self, epoch, total_step):
        self.model.eval()
        if dist.get_rank() == 0:
            v_pbar = tqdm(self.valid_data_loader, leave=False)
            tensor_to_pil = torchvision.transforms.ToPILImage()
            for step, test_data in enumerate(v_pbar):
                if step > 4:
                    break
                target_image, style_ref, mask_images, content_ref, style, target_image_name, origin_img_width, style_image_width = test_data['target_image'].to(self.device), \
                    test_data['style_image'].to(self.device), \
                    test_data['mask_image'].to(self.device), \
                    test_data['content_image'].to(self.device), \
                    test_data['style'], \
                    test_data['target_image_name'], \
                    test_data['origin_img_width'], \
                    test_data['style_image_width']
                
                batch_size = content_ref.shape[0]
                x = torch.randn((batch_size, 16, content_ref.shape[2]//8, content_ref.shape[3]//8)).to(self.device)
                ema_sampled_images = self.diffusion.ddim_sample(self.model, self.vae, batch_size, x, style_ref, content_ref)
                out_path = os.path.join(self.save_sample_dir, f"epoch_{epoch}")
                os.makedirs(out_path, exist_ok=True)

                for index in range(len(ema_sampled_images)):
                    s_img_width = style_image_width[index]
                    o_img_width = origin_img_width[index]

                    im = tensor_to_pil(ema_sampled_images[index])
                    image = im.convert("L")
                    style_path = os.path.join(out_path, style[index])
                    os.makedirs(style_path, exist_ok=True)
                    image = image.crop((0, 0, o_img_width, self.cfg.dataset.height))
                    image.save(os.path.join(style_path, target_image_name[index] + "_predict.png"))

                    t_im = target_image[index]
                    t_im = (t_im / 2 + 0.5).clamp(0, 1)
                    t_im = tensor_to_pil(t_im).convert("L")
                    t_im = t_im.crop((0, 0, o_img_width, self.cfg.dataset.height))
                    t_im.save(os.path.join(style_path, target_image_name[index] + "_target.png"))

                    c_im = content_ref[index]
                    c_im = (c_im / 2 + 0.5).clamp(0, 1)
                    c_im = tensor_to_pil(c_im).convert("L")
                    c_im = c_im.crop((0, 0, o_img_width, self.cfg.dataset.height))
                    c_im.save(os.path.join(style_path, target_image_name[index] + "_content.png"))

                    s_im = style_ref[index]
                    s_im = (s_im / 2 + 0.5).clamp(0, 1)
                    s_im = tensor_to_pil(s_im).convert("L")
                    s_im = s_im.crop((0, 0, s_img_width, self.cfg.dataset.height))
                    s_im.save(os.path.join(style_path, target_image_name[index] + "_style.png"))
                    
                del test_data, ema_sampled_images
            v_pbar.close()

    def train(self):
        """start training iterations"""
        for epoch in range(self.cfg.solver.epochs):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False)
            else:
                pbar = self.data_loader

            total_step = epoch * len(self.data_loader) 
            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.fine_tune:
                    self._finetune_iter(data, total_step, pbar)
                else:
                    self._train_iter(data, total_step, pbar)

            if (epoch+1) > self.cfg.train.snapshot_begin and (epoch+1) % self.cfg.train.snapshot_iters == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch,total_step)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > self.cfg.train.validate_begin  and (epoch+1) % self.cfg.train.validate_iters == 0:
                    self._valid_iter(epoch,total_step)
                else:
                    pass

            if dist.get_rank() == 0:
                pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch,total_step):
        checkpoint = {
            'epoch': epoch,  # 当前 epoch
            'total_step': total_step,
            'model_state_dict': self.model.module.state_dict(),  # 模型的参数 (DDP模型)
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器的参数
        }
        torch.save(checkpoint, os.path.join(self.save_model_dir, str(epoch)+'-'+"ckpt.pt"))