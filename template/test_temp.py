import os
import argparse
import torch
import torch.nn as nn
from dataloader.dataset import  BBXDataset
from omegaconf import OmegaConf
from utils.util import fix_seed
from utils.logger import set_log
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler 
from dataloader.collate_fn import CollateFN
from tqdm import tqdm
from torch import optim
from model.model import Temp
from trainer.trainer import Trainer
import logging
import torch.nn.functional as F
from datetime import datetime
from model.model import CFM_Wrapper

def main(opt):
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)

    fix_seed(cfg.train.seed)
    # logs = set_log(opt.output_dir, opt.cfg_file, opt.log_name)

    # import pdb;pdb.set_trace()

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

    test_bbx_dataset = BBXDataset(
        args=cfg.dataset,
        mode='ch', 
        phase='test'
    )
    test_sampler = DistributedSampler(test_bbx_dataset,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_bbx_dataset,
                                               batch_size=cfg.test.imgs_per_batch,
                                               drop_last=False,
                                               collate_fn=CollateFN(),
                                               num_workers=cfg.dataloader.num_threads,
                                               pin_memory=True,
                                               sampler=test_sampler)

    model = CFM_Wrapper(vocab_size=test_bbx_dataset.vocab_num+1).to(device)
    model = DDP(model, device_ids=[local_rank])
    ckpt_list = list(os.listdir(opt.pretrain))
    ckpt_list.sort(key=lambda x: int(x.split("-")[0]), reverse=True)
    
    log_dir = f'./metric'
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    log_file = os.path.join(log_dir, f'output_{current_time}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到终端
        ]
    )
    with torch.no_grad():
        for ckpt_name in ckpt_list:
            epoch = ckpt_name.split("-")[0]
            epoch_path = os.path.join(opt.pretrain, ckpt_name)
            checkpoint = torch.load(epoch_path, map_location="cpu")
            model.module.load_state_dict(checkpoint['model_state_dict'])
            print('load pretrained pretrain model from {}'.format(opt.pretrain))
            del checkpoint
            
            model.eval()
            v_pbar = tqdm(test_loader, leave=False)

            torch.set_printoptions(precision=8)
            loss_sum = torch.zeros(4, device=device)
            mask_sum = torch.zeros(1, device=device)
            loss_sum2 = []
            for step, test_data in enumerate(v_pbar):
                target_location, target_char_id, cond_mask, pred_mask, image_info, max_width = test_data['line_loc'].to(device), \
                        test_data['line_id'].to(device), \
                        test_data['cond_mask'].to(device), \
                        test_data['pred_mask'].to(device), \
                        test_data['image_info'], \
                        test_data['img_width']
                pred = model.module.inference(cond = target_location, char_ids = target_char_id , target = target_location, pred_mask = pred_mask,  cond_mask=cond_mask, n_timesteps=50)          

                loss = F.l1_loss(pred, target_location, reduction = 'none')
                loss = loss * pred_mask.unsqueeze(-1)

                loss_sum += loss.sum(dim=[0,1])
                mask_sum += pred_mask.sum()

                num = loss.sum(dim=1)
                den = pred_mask.sum(dim = -1).clamp(min = 1e-5)
                batch_loss = num / den.unsqueeze(-1)
                loss_sum2.append(batch_loss)
                del test_data
            
            loss_mean = loss_sum / mask_sum
            loss_sum2 = torch.cat(loss_sum2, dim=0)
            loss_mean2 = loss_sum2.mean(dim=0)
            final_loss = loss_mean.mean()
            logging.info(f'Epoch: {epoch} Loss: {loss_mean} final_loss: {final_loss} Loss2: {loss_mean2}')
            v_pbar.close()
            
        
if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', dest='pretrain', default='', help='pre-trained model')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/temp_config.yaml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log_name', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--output', dest='output_dir', default='./temp_metric',
                        help='save logs and checkpoints')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)
  


