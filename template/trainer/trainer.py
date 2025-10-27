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
from utils.util import make_template,recover_location

class Trainer:
    def __init__(self, cfg, model, optimizer, data_loader, 
                logs, id2char, valid_data_loader=None, device=None):
        self.cfg = cfg 
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.device = device
        self.id2char = id2char
      
    def _train_iter(self, data, step, pbar):
        self.model.train()
        target_location, target_char_id, cond_mask, pred_mask, img_info_list = data['line_loc'].to(self.device), \
                data['line_id'].to(self.device), \
                data['cond_mask'].to(self.device), \
                data['pred_mask'].to(self.device), \
                data['image_info']

        four_loss = self.model(cond = target_location, target = target_location, char_ids = target_char_id,  pred_mask = pred_mask,  cond_mask=cond_mask)
        loss = four_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"loss":loss.item(),"x0_loss":four_loss[0].item(), "y0_loss":four_loss[1].item(), "x1_loss":four_loss[2].item(),"y1_loss":four_loss[3].item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _valid_iter(self, epoch):
        
        self.model.eval()
        if dist.get_rank() == 0:
            v_pbar = tqdm(self.valid_data_loader, leave=False)
            for step, test_data in enumerate(v_pbar):
                if step > 1:
                    break
                target_location, target_char_id, cond_mask, pred_mask, image_info, max_width = test_data['line_loc'].to(self.device), \
                        test_data['line_id'].to(self.device), \
                        test_data['cond_mask'].to(self.device), \
                        test_data['pred_mask'].to(self.device), \
                        test_data['image_info'], \
                        test_data['img_width']

                pred = self.model.module.inference(cond = target_location, char_ids = target_char_id , target = target_location, pred_mask = pred_mask,  cond_mask=cond_mask, n_timesteps=100)  
                out_path = os.path.join(self.save_sample_dir, f"epoch_{epoch}")
                os.makedirs(out_path, exist_ok=True)
                for i in range(pred.shape[0]):
                    pred_location = pred[i][pred_mask[i].bool()]
                    id_list = target_char_id[i][pred_mask[i].bool()]
                    tgt_location = target_location[i][pred_mask[i].bool()]
                    img_width = max_width[i]

                    pred_location = pred_location.detach().cpu().numpy()
                    tgt_location = tgt_location.detach().cpu().numpy()
                    id_list = id_list.detach().cpu().numpy()

                    canvas_h, canvas_w = recover_location(location=tgt_location, 
                                    char_id=id_list, 
                                    img_width=img_width, 
                                    id2char=self.id2char, 
                                    img_info=image_info[i], 
                                    out_path=out_path,
                                    label = "target")
                    _ , _= recover_location(location=pred_location, 
                                    char_id=id_list, 
                                    img_width=img_width, 
                                    id2char=self.id2char, 
                                    img_info=image_info[i], 
                                    out_path=out_path,
                                    label = "pred",
                                    canvas_height = canvas_h,
                                    canvas_width = canvas_w)
                del test_data
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
                self._train_iter(data, total_step, pbar)

            if (epoch+1) > self.cfg.train.snapshot_begin and (epoch+1) % self.cfg.train.snapshot_iters == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch,total_step)
                else:
                    pass

            if self.valid_data_loader is not None:
                if (epoch+1) > self.cfg.train.validate_begin  and (epoch+1) % self.cfg.train.validate_iters == 0:
                    self._valid_iter(epoch)
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