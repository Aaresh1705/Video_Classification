import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import models

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
                super(DualStreamModel, self).__init__()
                self.num_classes = num_classes
                self.name = 'Dual Stream Model'
                self.height = 64
                self.width = 64
                # Frame stream (spatial)
                self.frame_conv = nn.Sequential(
                    # Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), 
                    nn.BatchNorm2d(96),  
                    nn.ReLU(),
                    nn.MaxPool2d(2), 

                    # Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), 
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # 16->8->4

                    # Conv3
                    nn.Conv2d(256, 512, 3, padding=1), 
                    nn.ReLU(),
                    
                    # Conv4
                    nn.Conv2d(512, 512, 3, padding=1),  
                    nn.ReLU(),

                    # Conv5
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    
                    # Conv6
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),  
                )
                
                # Frame stream fully connected layers
                self.frame_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 2 * 2, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
                
                # Flow stream (temporal) - similar architecture
                self.flow_conv = nn.Sequential(
                    # Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), 
                    nn.BatchNorm2d(96), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), 
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Conv3
                    nn.Conv2d(256, 512, 3, padding=1), 
                    nn.ReLU(),
                    
                    # Conv4
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(),

                    # Conv5
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    
                    # Conv6
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                # Flow stream fully connected layers
                self.flow_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 2 * 2, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
                
                # Late fusion classifier (combines both streams)
                self.fusion_classifier = nn.Sequential(
                    nn.Linear(2048 * 2, 512),  # Concatenate both 2048-dim features
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
                
            


            def forward(self, image_flow_batch):
                """
                Args:
                    image_flow_batch: tuple of (image_batch, flow_batch)
                        image_batch: [batch, channels, frames, height, width]
                        flow_batch: [batch, channels, frames, height, width]
                
                Returns:
                    output: [batch, num_classes] classification logits
                """
                (image_batch, flow_batch) = image_flow_batch
                batch_size, channels, frames, height, width = image_batch.shape
                
                # Process each frame through the spatial stream
                frame_features = []
                for t in range(frames):
                    frame = image_batch[:, :, t, :, :]  # [B, C, H, W]
                    conv_out = self.frame_conv(frame)    # [B, 512, 2, 2]
                    fc_out = self.frame_fc(conv_out)     # [B, 2048]
                    frame_features.append(fc_out)
                
                # Process each flow frame through the temporal stream
                flow_features = []
                for t in range(flow_batch.shape[2]):
                    flow = flow_batch[:, :, t, :, :]     # [B, C, H, W]
                    conv_out = self.flow_conv(flow)      # [B, 512, 2, 2]
                    fc_out = self.flow_fc(conv_out)      # [B, 2048]
                    flow_features.append(fc_out)
                
                # Aggregate features across time (average pooling)
                frame_features = torch.stack(frame_features, dim=1)  # [B, T, 2048]
                flow_features = torch.stack(flow_features, dim=1)    # [B, T, 2048]
                
                frame_agg = frame_features.mean(dim=1)  # [B, 2048]
                flow_agg = flow_features.mean(dim=1)    # [B, 2048]
                
                # Late fusion: concatenate both streams
                fused = torch.cat([frame_agg, flow_agg], dim=1)  # [B, 4096]
                
                # Final classification
                output = self.fusion_classifier(fused)  # [B, num_classes]
            
                return output
            
    model = DualStreamModel()
    return model


def get_simple_dualstream_model():
    
    class DualStreamModel(nn.Module):
            def __init__(self, num_classes=10):
                super(DualStreamModel, self).__init__()
                self.num_classes = num_classes
                self.name = 'Dual Stream Model'
                self.height = 64
                self.width = 64
                # Frame stream (spatial)
                self.frame_conv = nn.Sequential(
                    # Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), 
                    nn.BatchNorm2d(96),  
                    nn.ReLU(),
                    nn.MaxPool2d(2), 

                    # Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), 
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # 16->8->4

                    # Conv3
                    nn.Conv2d(256, 512, 3, padding=1), 
                    nn.ReLU(),
                    

                    # Conv4
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),  
                )
                
                # Frame stream fully connected layers
                self.frame_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 2 * 2, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
                
                # Flow stream (temporal) - similar architecture
                self.flow_conv = nn.Sequential(
                    # Conv1
                    nn.Conv2d(3, 96, 7, padding=3, stride=2), 
                    nn.BatchNorm2d(96), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Conv2
                    nn.Conv2d(96, 256, 5, padding=2, stride=2), 
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Conv3
                    nn.Conv2d(256, 512, 3, padding=1), 
                    nn.ReLU(),
                    

                    # Conv4
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                # Flow stream fully connected layers
                self.flow_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 2 * 2, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, 2048), 
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
                
                # Late fusion classifier (combines both streams)
                self.fusion_classifier = nn.Sequential(
                    nn.Linear(2048 * 2, 512),  # Concatenate both 2048-dim features
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
                
            
            def forward(self, image_flow_batch):
                """
                Args:
                    image_flow_batch: tuple of (image_batch, flow_batch)
                        image_batch: [batch, channels, frames, height, width]
                        flow_batch: [batch, channels, frames, height, width]
                
                Returns:
                    output: [batch, num_classes] classification logits
                """
                (image_batch, flow_batch) = image_flow_batch
                batch_size, channels, frames, height, width = image_batch.shape
                
                # Process each frame through the spatial stream
                frame_features = []
                for t in range(frames):
                    frame = image_batch[:, :, t, :, :]  # [B, C, H, W]
                    conv_out = self.frame_conv(frame)    # [B, 512, 2, 2]
                    fc_out = self.frame_fc(conv_out)     # [B, 2048]
                    frame_features.append(fc_out)
                
                # Process each flow frame through the temporal stream
                flow_features = []
                for t in range(flow_batch.shape[2]):
                    flow = flow_batch[:, :, t, :, :]     # [B, C, H, W]
                    conv_out = self.flow_conv(flow)      # [B, 512, 2, 2]
                    fc_out = self.flow_fc(conv_out)      # [B, 2048]
                    flow_features.append(fc_out)
                
                # Aggregate features across time (average pooling)
                frame_features = torch.stack(frame_features, dim=1)  # [B, T, 2048]
                flow_features = torch.stack(flow_features, dim=1)    # [B, T, 2048]
                
                frame_agg = frame_features.mean(dim=1)  # [B, 2048]
                flow_agg = flow_features.mean(dim=1)    # [B, 2048]
                
                # Late fusion: concatenate both streams
                fused = torch.cat([frame_agg, flow_agg], dim=1)  # [B, 4096]
                
                # Final classification
                output = self.fusion_classifier(fused)  # [B, num_classes]
            
                return output
            
    model = DualStreamModel()
    return model

# def get_pretrained_dualstream_model():
#     class PretrainedDualStreamModel(nn.Module):
#         def __init__(self, num_classes=10, freeze_backbones=True, pretrained=True):
#             super(PretrainedDualStreamModel, self).__init__()
#             self.name = 'Pretrained Dual Stream Model'
#             # --- Spatial Stream (RGB) ---
#             self.spatial_stream = models.resnet18(
#                 weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#             )
#             self.spatial_stream.fc = nn.Identity()  # remove classifier head

#             # --- Temporal Stream (Optical Flow) ---
#             self.temporal_stream = models.resnet18(
#                 weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#             )
#             # Modify first conv layer to accept 2-channel input (u,v)
#             self.temporal_stream.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             self.temporal_stream.fc = nn.Identity()

#             # --- Optional: Freeze early layers to avoid overfitting ---
#             if freeze_backbones:
#                 for param in self.spatial_stream.parameters():
#                     param.requires_grad = False
#                 for param in self.temporal_stream.parameters():
#                     param.requires_grad = False

#             # --- Fusion and Classification Head ---
#             self.fc = nn.Sequential(
#                 nn.Linear(512 * 2, 256),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(256, num_classes)
#             )

#         def forward(self, rgb, flow):
#             # Extract features from each stream
#             spatial_features = self.spatial_stream(rgb)
#             temporal_features = self.temporal_stream(flow)
#             # Concatenate feature vectors
#             combined = torch.cat((spatial_features, temporal_features), dim=1)
#             # Classification head
#             return self.fc(combined)

#     model = PretrainedDualStreamModel()
#     return model


def get_two_stream_dualstream_model():
    class TwoStreamSimonyanZisserman(nn.Module):
        def __init__(self, num_classes=101):
            super(TwoStreamSimonyanZisserman, self).__init__()
            self.num_classes = num_classes
            self.name = 'Two-Stream Simonyan & Zisserman (2014)'
            self.height = 64
            self.width = 64

            # ===== Spatial Stream (RGB frames) =====
            self.frame_conv = nn.Sequential(
                # conv1
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                # conv2
                nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                # conv3
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )

            self.frame_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 2 * 2, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

            # ===== Temporal Stream (Optical Flow) =====
            # Matches Simonyan & Zisserman flow architecture but scaled for 64×64
            self.flow_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )

            self.flow_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 2 * 2, 32),
                nn.ReLU(),
                nn.Dropout(0.9),
            )

            # ===== Late Fusion Classifier =====
            self.fusion_classifier = nn.Sequential(
                nn.Linear(32 * 2, num_classes),
            )

        def forward(self, image_flow_batch):
            """
            Args:
                image_flow_batch: tuple of (image_batch, flow_batch)
                    image_batch: [B, 3, T, 64, 64]
                    flow_batch:  [B, 2, T, 64, 64]
            Returns:
                output: [B, num_classes]
            """
            image_batch, flow_batch = image_flow_batch
            batch_size, channels, frames, height, width = image_batch.shape

            # --- Spatial Stream ---
            frame_features = []
            for t in range(frames):
                frame = image_batch[:, :, t, :, :]
                conv_out = self.frame_conv(frame)
                fc_out = self.frame_fc(conv_out)
                frame_features.append(fc_out)
            frame_agg = torch.stack(frame_features, dim=1).mean(dim=1)  # [B, 2048]

            # --- Temporal Stream ---
            flow_features = []
            for t in range(flow_batch.shape[2]):
                flow = flow_batch[:, :, t, :, :]
                conv_out = self.flow_conv(flow)
                fc_out = self.flow_fc(conv_out)
                flow_features.append(fc_out)
            flow_agg = torch.stack(flow_features, dim=1).mean(dim=1)  # [B, 2048]

            # --- Late Fusion ---
            fused = torch.cat([frame_agg, flow_agg], dim=1)  # [B, 4096]
            output = self.fusion_classifier(fused)           # [B, num_classes]
            return output



    model = TwoStreamSimonyanZisserman()
    return model
