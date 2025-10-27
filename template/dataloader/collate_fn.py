import torch
import torch.nn.functional as F

class CollateFN(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batched_data = {}
        batched_data["img_width"] = [ele["img_width"] for ele in batch]
        batched_data["image_info"] = [ele["image_info"] for ele in batch]
        line_len_list = [len(ele["line_id_list"]) for ele in batch] 
        max_len = max(line_len_list)

        line_id_list = []
        line_loc_list = []
        pred_mask_list = []
        cond_mask_list = []
        for ele in batch:
            line_loc = ele["line_location_list"]
            line_id = ele["line_id_list"]
            line_mask =  ele["line_mask"]
            seq_len = len(line_loc)
            padding_size = max_len - seq_len
            line_loc = line_loc + [[0,0,0,0]] * padding_size
            line_id = line_id + [0] * padding_size
            pred_mask = line_mask + [0] * padding_size
            cond_mask = [1 - x for x in line_mask] + [0] * padding_size

            line_loc_list.append(line_loc)
            line_id_list.append(line_id)
            pred_mask_list.append(pred_mask)
            cond_mask_list.append(cond_mask)

        line_loc_tensor = torch.tensor(line_loc_list)
        if batch[0]["phase"] ==  "train":
            line_loc_tensor += torch.randn_like(line_loc_tensor) * 0.001
        batched_data["line_loc"] = line_loc_tensor
        batched_data["line_id"] = torch.tensor(line_id_list)
        batched_data["pred_mask"] = torch.tensor(pred_mask_list)
        batched_data["cond_mask"] = torch.tensor(cond_mask_list)

        return batched_data
