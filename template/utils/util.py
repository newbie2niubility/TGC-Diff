import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import torch
import numpy as np
import random
import pygame
from PIL import Image, ImageDraw
import cv2
def load_ttf(ttf_path, fsize=96):
    pygame.init()
    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, fsize):
    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    imo = cv2.resize(imo, (fsize[1], fsize[0]))
    imo = imo.astype('uint8')
    # pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
    
    return imo

    
# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)

### model loads specific parameters (i.e., par) from pretrained_model 
def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def make_template(sample,target,char_id,mask,h,w,target_img_name,label):
    font = load_ttf("/mnt/wanghonglie/simhei.ttf")
    
    bs,_,_ = sample.shape
    for i in range(bs):
        img_name = target_img_name[i]
        white_image = Image.new('RGB', (w,h), color='white') 
        white_image_trg = Image.new('RGB', (w,h), color='white') 
        p_loc = sample[i].tolist()
        t_loc = target[i].tolist()
        id_list = char_id[i].tolist()
        m = mask[i].tolist()
        l = label[i]
        left_top_x = 0
        for j in range(sum(m)):
            [w1,w2,h1,h2] = p_loc[j]
            height = int((1-h2-h1) * h)
            width = int(w1 * h)
            font_id = l[j]
            char_image = ttf2im(font=font, char=font_id, fsize=[height,width])
            pil_im = Image.fromarray(char_image.astype('uint8')).convert('RGB')
            white_image.paste(pil_im, (left_top_x, int(h1*h)))
            left_top_x += int(w2 * h) + width
        white_image.save(img_name + '_pred2.png') 

        left_top_x = 0
        for j in range(sum(m)):
            [w1,w2,h1,h2] = t_loc[j]
            height = int((1-h2-h1) * h)
            width = int(w1 * h)
            font_id = l[j]
            char_image = ttf2im(font=font, char=font_id, fsize=[height,width])
            pil_im = Image.fromarray(char_image.astype('uint8')).convert('RGB')
            white_image_trg.paste(pil_im, (left_top_x, int(h1*h)))
            left_top_x += int(w2 * h) + width
        white_image_trg.save(img_name + '_trg2.png') 
        # import pdb;pdb.set_trace()


def recover_location(location, char_id, img_width, id2char, img_info, out_path, label, canvas_height=None, canvas_width = None):
    font = load_ttf("./font/simhei.ttf")
    para_name = img_info[0]
    writer = para_name.split('-')[0]
    line_index = img_info[1] + 1
    
    # int loc
    x0_list = [int(loc[0] * img_width) for loc in location]
    y0_list = [int(loc[1] * img_width) for loc in location]
    w_list = [int(loc[2] * img_width) for loc in location]
    h_list = [int(loc[3] * img_width) for loc in location]
    min_y0 = min(y0_list)
    min_x0 = min(x0_list)
    y0_list = [y0 - min_y0 for y0 in y0_list]
    x0_list = [x0 - min_x0 for x0 in x0_list]
    sum_list = [y0 + h for y0, h in zip(y0_list, h_list)]
    max_y1 = max(sum_list)
    
    # setting canvas
    if canvas_height == None:
        canvas_h = max_y1
    else:
        canvas_h = canvas_height
    if canvas_width == None: 
        width = sum(x0_list) + sum(w_list)
        canvas_w = max(width, img_width)
    else:
        canvas_w = canvas_width
    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

    # paste
    x_prev = 0
    for idx in range(len(location)):
        c_id = int(char_id[idx])
        font_id = id2char[c_id]
        x0, y0, w, h = x0_list[idx], y0_list[idx], w_list[idx], h_list[idx]
        x0 = x_prev + x0
        x_prev = x0 + w  
        try:
            char_image = ttf2im(font=font, char=font_id, fsize=[h,w])
            pil_im = Image.fromarray(char_image.astype('uint8')).convert('RGB')
            canvas.paste(pil_im, (x0, y0))
        except Exception as e:
            pass
        
    save_path = os.path.join(out_path, writer, para_name)
    os.makedirs(save_path, exist_ok=True)
    canvas.save(os.path.join(save_path, para_name + f'-L{line_index}_{label}' + '.png'))

    return  canvas_h, canvas_w   

