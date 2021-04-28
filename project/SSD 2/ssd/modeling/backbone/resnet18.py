import torch

from torchvision.models import resnet18

# RESNET18 300x300

def backbone_head_layer(channels_in, channels_out, strd, pad, last_layer=False):
    """
        Basic block appending to Resnet18 to get multiscale feature maps
    """
    if not last_layer:
        lr = torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(strd, strd), padding=(pad, pad), bias=False),
                torch.nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=(pad, pad), bias=False),
                torch.nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
    else:
        lr = torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(strd, strd), padding=(pad, pad), bias=False),
                torch.nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channels_out, channels_out, kernel_size=(1, 1), stride=(1, 1), padding=(pad, pad), bias=False),
                torch.nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
    
    return lr

class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        
        model = resnet18(pretrained=True)      
        
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]), 
                                                     backbone_head_layer(512,256,2,1), 
                                                     backbone_head_layer(256,256,2,1), 
                                                     backbone_head_layer(256,256,2,0,True))
        #print("layers:", self.feature_extractor)
        
                    

    def forward(self, features):
    
        features_out = []
        
        #resnet18
        features = self.feature_extractor[0](features)
        features = self.feature_extractor[1](features)
        features = self.feature_extractor[2](features)
        features = self.feature_extractor[3](features)
        features = self.feature_extractor[4](features)
        features = self.feature_extractor[5](features)
        features_out.append(features)
        features = self.feature_extractor[6](features)
        features_out.append(features)
        features = self.feature_extractor[7](features)
        features_out.append(features)
        
        # extra feature maps (5x5, 3x3 and 1x1)
        features = self.feature_extractor[8](features)        
        features_out.append(features)        
        features = self.feature_extractor[9](features)        
        features_out.append(features)        
        features = self.feature_extractor[10](features)     
        features_out.append(features)
        
        out = features_out
            
        return tuple(out)
