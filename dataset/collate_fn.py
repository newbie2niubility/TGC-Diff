import torch
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

def RightPadWhite(final_img_width, image, fill_number=255):
    fill_number = (fill_number,fill_number,fill_number)
    new_width, new_height = image.size
    padding_width = final_img_width - new_width
    if padding_width > 0:
        new_image = Image.new("RGB", (final_img_width, new_height), color=fill_number)
        new_image.paste(image, (0, 0))
    elif padding_width < 0:
        raise ValueError(f"Image width {new_width} is larger than final width {final_img_width}.")
    else:
        new_image = image
    return new_image

def RightPadPrev(target_width, image):
    new_width, target_height = image.size
    padding_width = target_width - new_width
    if padding_width > 0:
        new_image = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
        new_image.paste(image, (0, 0))
        current_x = new_width
        while current_x < target_width:
            remaining_width = target_width - current_x
            if remaining_width >= new_width:
                new_image.paste(image, (current_x, 0))
                current_x += new_width
            else:
                part = image.crop((0, 0, remaining_width, target_height))
                new_image.paste(part, (current_x, 0))
                current_x += remaining_width
    elif padding_width < 0:
        raise ValueError(f"Image width {new_width} is larger than final width {target_width}.")
    else:
        new_image = image
    return new_image


class CollateFN(object):
    def __init__(self):
        self.pil_to_tensor = T.Compose(
                    [T.ToTensor(),
                    T.Normalize([0.5], [0.5])])

    def __call__(self, batch):
        new_batch = {}
        new_batch["target_image_path"] = [ele["target_image_path"] for ele in batch]
        new_batch["style"] = [ele["style"] for ele in batch]
        new_batch["target_image_name"] = [ele["target_image_name"] for ele in batch]

        origin_width_list = [ele["origin_img_width"] for ele in batch]
        style_width_list = [ele["style_image_width"] for ele in batch]

        new_batch["origin_img_width"] = origin_width_list
        new_batch["style_image_width"] = style_width_list

        max_origin_width = max(origin_width_list)
        max_style_width = max(style_width_list)

        origin_image_list = [RightPadWhite(max_origin_width, ele["origin_image"])  for ele in batch]
        content_image_list = [RightPadWhite(max_origin_width, ele["content_image"])  for ele in batch]

        style_image_list = [RightPadPrev(max_style_width, ele["style_image"])  for ele in batch]

        origin_img_tensor = torch.stack([self.pil_to_tensor(image) for image in origin_image_list], dim=0)
        content_img_tensor = torch.stack([self.pil_to_tensor(image) for image in content_image_list], dim=0)
        style_img_tensor = torch.stack([self.pil_to_tensor(image) for image in style_image_list], dim=0)

        new_batch["target_image"] = origin_img_tensor
        new_batch["content_image"] = content_img_tensor
        new_batch["style_image"] = style_img_tensor

        img_tensor = [torch.from_numpy(np.array(image.convert("L"))).float() for image in origin_image_list]
        img_tensor = torch.stack(img_tensor, dim=0).unsqueeze(1)
        padded_image_tensor = F.pad(img_tensor, (1, 1, 1, 1), mode='replicate')
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        high_freq_features_x = F.conv2d(padded_image_tensor, sobel_kernel_x)
        high_freq_features_y = F.conv2d(padded_image_tensor, sobel_kernel_y)
        high_freq_features = torch.sqrt(high_freq_features_x ** 2 + high_freq_features_y ** 2)
        threshold = high_freq_features.mean() + 2 * high_freq_features.std()
        binary_high_freq_features = (high_freq_features > threshold).float()
        mask_image = 0.3*torch.ones_like(binary_high_freq_features) + 0.7*binary_high_freq_features
        new_batch["mask_image"] = mask_image

        return new_batch

# class CollateFN(object):
#     def __init__(self):
#         pass

#     def __call__(self, batch):
#         batched_data = {}
#         for k in batch[0].keys():
#             if k != "target" and k != "target_lengths":
#                 batch_key_data = [ele[k] for ele in batch]
#                 if isinstance(batch_key_data[0], torch.Tensor):
#                     batch_key_data = torch.stack(batch_key_data)
#                 batched_data[k] = batch_key_data

#         return batched_data
