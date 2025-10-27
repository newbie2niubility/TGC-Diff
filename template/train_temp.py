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
from model.model import CFM_Wrapper
from trainer.trainer import Trainer

def main(opt):
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)

    fix_seed(cfg.train.seed)
    logs = set_log(opt.output_dir, opt.cfg_file, opt.log_name)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

    train_bbx_dataset = BBXDataset(
        args=cfg.dataset,
        mode='ch', 
        phase='train'
    )
    train_sampler = DistributedSampler(train_bbx_dataset)
    train_loader = torch.utils.data.DataLoader(train_bbx_dataset,
                                               batch_size=cfg.train.imgs_per_batch,
                                               drop_last=False,
                                               collate_fn=CollateFN(),
                                               num_workers=cfg.dataloader.num_threads,
                                               pin_memory=True,
                                               sampler=train_sampler)

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
    
    model = CFM_Wrapper(vocab_size=train_bbx_dataset.vocab_num+1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.solver.base_lr)

    if len(opt.pretrain) > 0:
        checkpoint = torch.load(opt.pretrain, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('load pretrained pretrain model from {}'.format(opt.pretrain))
        del checkpoint

    # # total_params = sum(p.numel() for p in model.parameters())
    # # print(f"模型总参数量: {total_params}")
    # # import pdb;pdb.set_trace()
    model = DDP(model, device_ids=[local_rank])

    trainer = Trainer(cfg, model, optimizer, train_loader, logs, test_bbx_dataset.id2char, test_loader, device)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', dest='pretrain', default='', help='pre-trained model')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/temp_config.yaml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log_name', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--output', dest='output_dir', default='./output',
                        help='save logs and checkpoints')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)



