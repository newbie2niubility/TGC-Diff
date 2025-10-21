import os
import random
import json
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def RightPadWhite(target_height,target_width, image):
    original_width,  original_height = image.size
    new_width = int(original_width * (target_height / original_height))
    image_resized = image.resize((new_width, target_height))
    padding_width = target_width - new_width
    if padding_width > 0:
        new_image = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
        new_image.paste(image_resized, (0, 0))
    elif padding_width < 0:
        new_image = image_resized.crop((0, 0, target_width, target_height))
    else:
        new_image = image_resized.copy()
    return new_image

def RightPadPrev(target_height, target_width, image):
    original_width, original_height = image.size
    new_width = int(original_width * (target_height / original_height))
    image_resized = image.resize((new_width, target_height))
    padding_width = target_width - new_width
    if padding_width > 0:
        new_image = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
        new_image.paste(image_resized, (0, 0))
        current_x = new_width
        while current_x < target_width:
            remaining_width = target_width - current_x
            if remaining_width >= new_width:
                new_image.paste(image_resized, (current_x, 0))
                current_x += new_width
            else:
                part = image_resized.crop((0, 0, remaining_width, target_height))
                new_image.paste(part, (current_x, 0))
                current_x += remaining_width
    elif padding_width < 0:
        start_x = random.randint(0, new_width - target_width)
        new_image = image_resized.crop((start_x, 0, start_x + target_width, target_height))
    else:
        new_image = image_resized.copy()
    return new_image
    
class TextLineDataset(Dataset):
    def __init__(self, args, mode, phase):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.mode = mode
        self.target_width = args.width
        self.target_height = args.height
        self.reference_num = args.reference_num
        self.template_root = args.data_root + '_template'

        if mode == 'chinese':
            if self.phase == 'train':
                self.data_path = os.path.join(self.root,'train')
                writer_id_list = list(os.listdir(self.data_path))
                self.writer_id = sorted(writer_id_list, key=lambda x: int(x))
                self.writer_num = len(self.writer_id)
            if self.phase == 'test':
                self.data_path = os.path.join(self.root,'test')
                writer_id_list = list(os.listdir(self.data_path))
                self.writer_id = sorted(writer_id_list, key=lambda x: int(x[1:]))
                self.writer_num = len(self.writer_id)
        self.get_path()
    
    def get_path(self):
        self.target_images = []
        self.template_images = []
        self.style_images = {}

        self.style_list = []
        self.label_list = []
        self.id_list = []

        for style in self.writer_id:
            images_related_style = []
            passage_path = os.path.join(self.data_path,style)
            for passage in os.listdir(passage_path):
                line_path = os.path.join(passage_path,passage)
                for img in os.listdir(line_path):
                    img_path = os.path.join(line_path,img)
                    img_name = img.split('.')[0]
                    self.target_images.append(img_path)
                    template_path = os.path.join(self.template_root, self.phase, style, passage, img_name+"_template.png")
                    self.template_images.append(template_path)
                    images_related_style.append(img_path)
                    self.style_list.append(style)
            self.style_images[style] = images_related_style
        
    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        content_image_path = self.template_images[index]
        target_image_name = target_image_path.split('/')[-1].split('.')[0]
        style = self.style_list[index]

        origin_image = Image.open(target_image_path).convert("RGB")
        origin_image = RightPadWhite(self.target_height ,self.target_width, origin_image)

        content_image = Image.open(content_image_path).convert('RGB')
        content_image = RightPadWhite(self.target_height ,self.target_width, content_image)

        images_related_style = self.style_images[style].copy()
        if target_image_path in images_related_style:
            images_related_style.remove(target_image_path)
        else:
            raise ValueError(f"Error: {target_image_path} not found in images_related_style.")

        if self.reference_num == 1:
            style_image_path = random.choice(images_related_style)
        else:
            extended_images = images_related_style * ((self.reference_num // len(images_related_style)) + 1)
            style_image_path = random.sample(extended_images, self.reference_num)

        if isinstance(style_image_path, list):
            style_image = []
            for style_path in style_image_path:
                reference_image = Image.open(style_path).convert("RGB")
                style_image.append(RightPadPrev(self.target_height ,self.target_width,reference_image))
        else:
            style_image = Image.open(style_image_path).convert("RGB")
            style_image = RightPadPrev(self.target_height ,self.target_width,style_image)

        pil_to_tensor = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

        target_image = pil_to_tensor(origin_image)
        content_image = pil_to_tensor(content_image)
        style_image = pil_to_tensor(style_image)

        image = origin_image.convert("L")
        image_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0).unsqueeze(0)
        padded_image_tensor = F.pad(image_tensor, (1, 1, 1, 1), mode='replicate')
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        high_freq_features_x = F.conv2d(padded_image_tensor, sobel_kernel_x)
        high_freq_features_y = F.conv2d(padded_image_tensor, sobel_kernel_y)
        high_freq_features = torch.sqrt(high_freq_features_x ** 2 + high_freq_features_y ** 2)
        threshold = high_freq_features.mean() + 2 * high_freq_features.std()
        binary_high_freq_features = (high_freq_features > threshold).float()
        mask_image = 0.3*torch.ones_like(target_image[0]) + 0.7*binary_high_freq_features.squeeze(dim=0)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "mask_image":mask_image,
            "style":style,
            "target_image_name":target_image_name,
        }
        return sample

    def __len__(self):
        return len(self.target_images)










