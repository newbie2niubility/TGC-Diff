import torch
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from einops import rearrange

class StyleEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super(StyleEncoder, self).__init__()
        self.d_model = d_model
        self.resnet18 = CustomResNet18()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers, style_norm)

        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, style):
        origin_height = style.shape[2]
        style = self.resnet18(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=self.d_model, h = origin_height//16 ).contiguous()
        style = self.add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        style = self.transformer(style)
        return style.permute(1, 0, 2).contiguous()


class CustomResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomResNet18, self).__init__()
        
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT' if pretrained else None)
        
        for block in self.resnet.layer4:
            for name, module in block.named_children():
                if isinstance(module, nn.Conv2d):
                    module.stride = (1, 1) 
        
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        
    def forward(self, x):
        return self.resnet(x)
