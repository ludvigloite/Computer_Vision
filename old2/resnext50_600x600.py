import torch


from torchvision.models import resnext50_32x4d


def backbone_head_layer(channels_in, channels_out, strd, pad, last_layer=False):
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

class ResNeXt50_600x600(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        
        model = resnext50_32x4d(pretrained=True)      
        
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]), 
                                                     backbone_head_layer(2048,512,2,1), 
                                                     backbone_head_layer(512,512,2,1), 
                                                     backbone_head_layer(512,512,2,1), 
                                                     backbone_head_layer(512,512,2,0,True))
        print("layers:", self.feature_extractor)
        #print(torch.nn.Sequential(*(list(resnext_model.children()))))
        

        

                    

    def forward(self, features):
        """
        
        """
        features_out = []
        
        output_nr = [1,5,]
        
        #backbone
        features = self.feature_extractor[0](features)
        features = self.feature_extractor[1](features)
        features = self.feature_extractor[2](features)
        features = self.feature_extractor[3](features)
        features = self.feature_extractor[4](features)        
        features = self.feature_extractor[5](features)
        #features_out.append(features)
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
        
        features = self.feature_extractor[11](features)     
        features_out.append(features)
        
        
        out = features_out
        """
        for idx, feature in enumerate(out):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
            
        return tuple(out)
