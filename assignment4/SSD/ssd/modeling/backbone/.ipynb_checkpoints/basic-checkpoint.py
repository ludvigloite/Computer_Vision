import torch


class BasicModel(torch.nn.Module):
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
        
        model = 2
        
        if model == 1:
            self.feature_extractor_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                torch.nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                torch.nn.ReLU(),

                torch.nn.Conv2d(32, 64, 3, 1, 1),
                torch.nn.MaxPool2d(2,2),
                torch.nn.ReLU(),


                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, output_channels[0], 3, 2, 1),
            )

            self.feature_extractor_2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[0], 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[1], 3, 2, 1),
            )
            self.feature_extractor_3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[1], 256, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[2], 3, 2, 1),
            )
            self.feature_extractor_4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[2], 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], 3, 2, 1),
            )
            self.feature_extractor_5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[3], 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], 3, 2, 1),
            )
            self.feature_extractor_6 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[4], 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[5], 3, 1, 0),
            )
            
        elif model == 2:
            self.feature_extractor_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                #torch.nn.BatchNorm2d(32),
                torch.nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                torch.nn.ReLU(),

                torch.nn.Conv2d(32, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(2,2),
                torch.nn.ReLU(),


                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, output_channels[0], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[0]),
            )

            self.feature_extractor_2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[0], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[1], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[1]),
            )
            self.feature_extractor_3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[1], 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[2], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[2]),
            )
            self.feature_extractor_4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[2], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[3]),
            )
            self.feature_extractor_5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[3], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[4]),
            )
            self.feature_extractor_6 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[4], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[5], 3, 1, 0),
                #torch.nn.BatchNorm2d(output_channels[5]),
            )
            
            
        elif model==3:
            self.feature_extractor_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                torch.nn.BatchNorm2d(32),
                torch.nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                torch.nn.ReLU(),

                torch.nn.Conv2d(32, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(2,2),
                torch.nn.ReLU(),


                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),

                torch.nn.Conv2d(64, output_channels[0], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[0]),
            )

            self.feature_extractor_2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[0], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[1], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[1]),
            )
            self.feature_extractor_3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[1], 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),

                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[2], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[2]),
            )
            self.feature_extractor_4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[2], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[3]),
            )
            self.feature_extractor_5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[3], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], 3, 2, 1),
                #torch.nn.BatchNorm2d(output_channels[4]),
            )
            self.feature_extractor_6 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[4], 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),

                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[5], 3, 1, 0),
                #torch.nn.BatchNorm2d(output_channels[5]),
            )
        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        features_out = []
        batch_size = x.shape[0]
        features = self.feature_extractor_1(x)
        features_out.append(features)
        features = self.feature_extractor_2(features)
        features_out.append(features)
        features = self.feature_extractor_3(features)
        features_out.append(features)
        features = self.feature_extractor_4(features)
        features_out.append(features)
        features = self.feature_extractor_5(features)
        features_out.append(features)
        features = self.feature_extractor_6(features)
        features_out.append(features)
        
        out = features_out
        
        
        for idx, feature in enumerate(out):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
            
        return tuple(out)

