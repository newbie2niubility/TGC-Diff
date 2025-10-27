import random
from PIL import Image
import json
import os
import argparse
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from collections import defaultdict
import copy

class BBXDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, mode, phase):
        super().__init__()

        if mode == 'ch':
            with open(args.ch_vocab, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            self.id2char = {i + 1: char for i, char in enumerate(lines)}
            self.char2id = {char: i + 1 for i, char in enumerate(lines)}
            self.vocab_num = len(self.id2char.keys())
            self.mode = mode

            if phase=='train':
                with open(args.ch_train_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.phase = phase
            else:
                with open(args.ch_test_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.phase = phase

        self.data = {}
        self.data_list = []
        for img_name, img_info in data.items():
            char_location = data[img_name]['char_location']
            rec_location = data[img_name]['rec_location']
            para_size = data[img_name]['size']
            para_x0, para_y0 = para_size[:2]
            para_width = para_size[2] - para_size[0]
            para_height = para_size[3] - para_size[1]
            
            para_length = 0

            line_location_list = []
            line_id_list = []
            line_para_change_list = []
            old_rec_y1 = 0
            
            for i in range(len(char_location)):
                char_location_list =[]
                char_id_list = []
                rx0,ry0,rx1,ry1 = rec_location[i]
                ry0 = (ry0 - para_y0)/para_width
                ry1 = (ry1 - para_y0)/para_width
                old_x0 = 0
                line_delta_x0 = []
                for j, char_list in enumerate(char_location[i]):
                    char_id, [c_x0, c_y0, c_x1, c_y1] = char_list
                    if char_id not in self.char2id.keys():
                        char_id_list.append(7357)
                    else:    
                        char_id_list.append(self.char2id[char_id])
                    c_x0 = (c_x0 - para_x0)/para_width
                    c_y0 = (c_y0 - para_y0)/para_width
                    c_x1 = (c_x1 - para_x0)/para_width
                    c_y1 = (c_y1 - para_y0)/para_width
                    
                    delta_x0 = c_x0 - old_x0
                    delta_w = c_x1 - c_x0
                    old_x0 = c_x1
                    delta_y0 = c_y0 - old_rec_y1
                    delta_h = c_y1 - c_y0

                    if j != 0 :
                        line_delta_x0.append(delta_x0)
                    
                    char_location_list.append([delta_x0,delta_y0,delta_w,delta_h])
                
                if len(line_delta_x0)>0 :
                    char_location_list[0][0] = sum(line_delta_x0) / len(line_delta_x0)
     
                old_rec_y1 = ry1
                para_length += len(char_location[i])
                line_location_list.append(char_location_list)
                line_id_list.append(char_id_list)
                self.data_list.append((img_name, i))
            
            self.data[img_name] = {'line_location_list':line_location_list, 'line_id_list':line_id_list, 'img_width':para_width}
                
    def __getitem__(self, index):
        image_name, line_index = self.data_list[index]
        data = self.data[image_name]

        img_width = data['img_width']
        line_id_list = data['line_id_list']
        line_location_list = data['line_location_list']

        target_line_id = line_id_list[line_index]
        target_line_loc = line_location_list[line_index]
        if self.phase == "train":
            available_indices = list(range(len(line_id_list)))
            available_indices.remove(line_index)
            style_index = random.choice(available_indices)
            style_line_id = line_id_list[style_index]
            style_line_loc = line_location_list[style_index]
        else:
            if line_index == 0:
                style_line_id = line_id_list[1]
                style_line_loc = line_location_list[1]
            else:
                style_line_id = line_id_list[0]
                style_line_loc = line_location_list[0]

        combined_line_id = style_line_id + target_line_id
        combined_line_loc = style_line_loc + target_line_loc
        mask = [0]*len(style_line_id) + [1]*len(target_line_id)

        sample = {
            "line_location_list": combined_line_loc,
            "line_id_list": combined_line_id,
            "line_mask": mask,
            "img_width": img_width,
            "image_info":(image_name, line_index),
            "phase": self.phase
            }
        return sample
            
    def __len__(self):
        return len(self.data_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='./config/Chinese_template.yaml',
                        help='Config file for training (and optionally testing)')
    opt = parser.parse_args()
    cfg = OmegaConf.load(opt.cfg_file)
    OmegaConf.set_struct(cfg, True)
    train_bbx_dataset = BBXDataset(
        args=cfg.dataset,
        mode='trian', 
    )
