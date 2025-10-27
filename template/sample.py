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
from utils.util import recover_location

def main(opt):
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)

    fix_seed(cfg.train.seed)
    logs = set_log(opt.output_dir, opt.cfg_file, opt.log_name)

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
    
    if len(opt.pretrain) > 0:
        checkpoint = torch.load(opt.pretrain, map_location="cpu")
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print('load pretrained pretrain model from {}'.format(opt.pretrain))
        del checkpoint
    
    with torch.no_grad(): 
        model.eval()
        v_pbar = tqdm(test_loader, leave=False)

        for step, test_data in enumerate(v_pbar):
            target_location, target_char_id, cond_mask, pred_mask, image_info, max_width = test_data['line_loc'].to(device), \
                    test_data['line_id'].to(device), \
                    test_data['cond_mask'].to(device), \
                    test_data['pred_mask'].to(device), \
                    test_data['image_info'], \
                    test_data['img_width']
            pred = model.module.inference(cond = target_location, char_ids = target_char_id , target = target_location, pred_mask = pred_mask,  cond_mask=cond_mask, n_timesteps=50)          

            out_path = logs['sample']
            os.makedirs(out_path, exist_ok=True)
            for i in range(pred.shape[0]):
                pred_location = pred[i][pred_mask[i].bool()]
                id_list = target_char_id[i][pred_mask[i].bool()]
                tgt_location = target_location[i][pred_mask[i].bool()]
                img_width = max_width[i]

                pred_location = pred_location.detach().cpu().numpy()
                tgt_location = tgt_location.detach().cpu().numpy()
                id_list = id_list.detach().cpu().numpy()

                # canvas_h, canvas_w = recover_location(location=tgt_location, 
                #                 char_id=id_list, 
                #                 img_width=img_width, 
                #                 id2char=test_bbx_dataset.id2char, 
                #                 img_info=image_info[i], 
                #                 out_path=out_path,
                #                 label = "target")
                _ , _= recover_location(location=pred_location, 
                                char_id=id_list, 
                                img_width=img_width, 
                                id2char=test_bbx_dataset.id2char, 
                                img_info=image_info[i], 
                                out_path=out_path,
                                label = "pred")
            del test_data
        v_pbar.close()
        
        
        
if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', dest='pretrain', default='', help='pre-trained model')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/temp_config.yaml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log_name', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--output', dest='output_dir', default='./template/sample_vis',
                        help='save logs and checkpoints')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)
  


