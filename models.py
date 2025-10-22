import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights


def get_vgg16_model(pretrained: bool=False, custom_weights: str='') -> nn.Module:
    """
    Returns the VGG16 model
    :param pretrained: Loads the pretrained model if True
    :param custom_weights: If pretrained is False and custom_weights are given then they will be loaded into the model.
    :return: The VGG16 model
    """

    def get_model(weights) -> nn.Module:
        model = vgg16(weights=weights)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    if pretrained:
        model = get_model(weights=VGG16_Weights.DEFAULT)

        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = get_model(weights=None)

        for param in model.features.parameters():
            param.requires_grad = True

    if custom_weights != '':
        model.load_state_dict(torch.load(custom_weights, weights_only=True))
        print('Loaded custom weights:', custom_weights)

    return model


def get_resnet_model(pretrained: bool=False, custom_weights: str='', lock_feature_extractor=False) -> nn.Module:
    def get_model(weights) -> nn.Module:
        model = resnet50(weights=weights)
        model.fc.out_features = 10

        for param in model.parameters():
            param.requires_grad = True

        return model

    if pretrained:
        model = get_model(weights=ResNet50_Weights.DEFAULT)

        if lock_feature_extractor:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    else:
        model = get_model(weights=None)

    if custom_weights != '':
        model.load_state_dict(torch.load(custom_weights, weights_only=True))
        print('Loaded custom weights:', custom_weights)

    return model


def get_single_frame_model():
    class Network(nn.Module):
        def __init__(self, num_classes=10):
            super(Network, self).__init__()
            # implementing a single frame model for classification #data loader returns [8, 3, 64, 64]
            self.conv = nn.Sequential( # input is 64x64
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),   # 32x32

                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),   # 16x16

                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),   # 8x8

                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),  # new conv layer

                nn.AdaptiveAvgPool2d(1) # -> (256, 1, 1)
            )
            # fully connected part
            self.fc = nn.Sequential(
                nn.Linear(256, 128),  # new hidden layer
                nn.ReLU(),
                nn.Dropout(0.5),      # to avoid overfitting
                nn.Linear(128, num_classes)
            )


        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)  # flatten
            return self.fc(x)

    model = Network()

    return model


def get_late_fusion_model():
    class LateFusion(nn.Module):
        def __init__(self, num_classes=10, name='Late Fusion'):
            super(LateFusion, self).__init__()

            self.name = name
            self.num_classes = num_classes

        def forward_each_frame(self, batch, num_frames):
            # process each frame independently through the conv layers
            frame_features = []
            for f in range(num_frames):
                frame = batch[:, :, f, :, :]  # [B, 3, H, W]
                x = self.feature_extractor(frame)  # [B, D, H', W']
                x = x.flatten(start_dim=1)  # [B, D * H' * W']
                frame_features.append(x)  # [T, B, D * H' * W']

            return frame_features


    class LateFusionModelMLP(LateFusion):
        def __init__(self, num_classes=10):
            super().__init__(num_classes=num_classes, name='Late Fusion MLP')

            # shared feature extractor for each frame
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 32x32

                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8

                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Dropout(0.3),
            )

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8 * 10, 256), # 64 * 8 * 8 * 10
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, image_batch):
            # image_batch: [batch, channels, frames, height, width]
            batch_size, channels, frames, height, width = image_batch.shape

            frame_features = self.forward_each_frame(image_batch, frames)

            x = torch.stack(frame_features, dim=1)  # [B, T, D * H' * W']
            x = x.flatten(start_dim=1)  # [B, T * D * H' * W']
            y = self.classifier(x)
            return y


    class LateFusionModelResNetMLP(LateFusion):
        def __init__(self, num_classes=10):
            super().__init__(num_classes=num_classes, name='Late Fusion ResNet MLP')
            resnet = get_resnet_model(pretrained=True)
            # shared feature extractor for each frame
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(2048 * 2 * 2 * 10, 256), # 2048 * 2 * 2 * 10 ; 512 * 2 * 2 * 10
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, image_batch):
            # image_batch: [batch, channels, frames, height, width]
            batch_size, channels, frames, height, width = image_batch.shape

            # process each frame independently through the conv layers
            frame_features = self.forward_each_frame(image_batch, frames)

            x = torch.stack(frame_features, dim=1)  # [B, T, D * H' * W']
            x = x.flatten(start_dim=1)              # [B, T * D * H' * W']
            y = self.classifier(x)
            return y


    class LateFusionModelPooling(LateFusion):
        def __init__(self, num_classes=10):
            super().__init__(num_classes=num_classes, name='Late Fusion Pooling')

            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 32x32

                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8

                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Dropout(0.3),

                nn.AdaptiveAvgPool2d(1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(128, 10)
            )

        def forward(self, image_batch):
            # image_batch: [batch, channels, frames, height, width]
            batch_size, channels, frames, height, width = image_batch.shape

            frame_features = self.forward_each_frame(image_batch, frames)

            x = torch.stack(frame_features, dim=1)  # [B, T, D]
            x = x.mean(dim=1)          # [B, D]

            y = self.classifier(x)
            return y


    class LateFusionModelPoolingResNet(LateFusion):
        def __init__(self, num_classes=10):
            super().__init__(num_classes=num_classes, name='Late Fusion ResNet POOLING')

            # shared feature extractor for each frame
            resnet = get_resnet_model(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

            self.classifier = nn.Sequential(
                nn.Linear(2048, 10),    # 2048 ; 512
            )

        def forward(self, image_batch):
            # image_batch: [batch, channels, frames, height, width]
            batch_size, channels, frames, height, width = image_batch.shape

            frame_features = self.forward_each_frame(image_batch, frames)

            x = torch.stack(frame_features, dim=1)  # [B, T, D]
            x = x.mean(dim=1)          # [B, D]

            y = self.classifier(x)
            return y


    mlp = LateFusionModelMLP()
    pooling = LateFusionModelPooling()

    mlp_resnet = LateFusionModelResNetMLP()
    pooling_resnet = LateFusionModelPoolingResNet()

    return mlp, pooling, mlp_resnet, pooling_resnet

def get_video_frame_model():
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            # implementing a single frame model for classification


        def forward(self, x):
            ...
            return x

    model = Network()

    return model

def get_dualstream_model():
    
    class DualStreamModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__(num_classes=num_classes, name='Dual Stream Model')
                height = 64
                width = 64
                self.frame_model = nn.Sequential(
                    #Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(2),  # 16x16 due to strid + pool

                    #Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(2),  # 4x4 due to stride + pool

                    #Conv3
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    
                    #Conv4
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),

                    #Conv4
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    
                    
                    #Conv5
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),  # 2x2 due to stride + pool
                    
                    #full6
                    nn.Linear(512 * 2 * 2, 2048), nn.ReLU(), # (C * H * W)
                    nn.Dropout(),
                    
                    #full7
                    nn.Linear(2048, 2048), nn.ReLU(), # (C * H * W)
                    nn.Dropout(),
                    
                    #classifier
                    nn.Linear(2048,10), 
                    
                    nn.Softmax(dim=1)
                    
                )
            
            
                self.flow_model = nn.Sequential(
                    #Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(2),  # 16x16 due to strid + pool

                    #Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), nn.ReLU(),
                    nn.MaxPool2d(2),  # 4x4 due to stride + pool

                    #Conv3
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    
                    #Conv4
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),

                    #Conv4
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    
                    
                    #Conv5
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),  # 2x2 due to stride + pool
                    
                    #full6
                    nn.Linear(512 * 2 * 2, 2048), nn.ReLU(), # (C * H * W)
                    nn.Dropout(),
                    
                    #full7
                    nn.Linear(2048, 2048), nn.ReLU(), # (C * H * W)
                    nn.Dropout(),
                    
                    #classifier
                    nn.Linear(2048,10), 
                    
                    nn.Softmax(dim=1)
                    
                )
                
            


            def forward(self, image_flow_batch):
                # image_batch: [batch, channels, frames, height, width]
                image_batch, flow_batch = image_flow_batch
                batch_size, channels, frames, height, width = image_batch.shape
                
                
                flow_batch_size, flow_channels, flow_frames, flow_height, flow_width = flow_batch.shape

                
                return 