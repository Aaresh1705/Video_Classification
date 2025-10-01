import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights


def get_vgg16_model(pretrained: bool=False, custom_weights: str='') -> nn.Module:
    """
    Returns the VGG16 model
    :param pretrained: Loads the pretrained model if True
    :param custom_weights: If pretrained is False and custom_weights are given then they will be loaded into the model.
    :return: The VGG16 model
    """

    def get_model(weights) -> nn.Module:
        model = vgg16(weights=weights)
        # print(model.classifier)
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


def get_Single_Frame_model():
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

def get_Video_Frame_model():
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            # implementing a single frame model for classification


        def forward(self, x):
            ...
            return x

    model = Network()

    return model
