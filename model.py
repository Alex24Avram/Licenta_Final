import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


#Model CNN-4L (4 conv + 3 fc)
class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.flatten_dim = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_speakers)

    def _get_flatten_size(self):
        x = torch.zeros(1, 1, 128, 200)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def extract_embedding(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x



# Model CNN-3L (3 conv + 2 fc)

class SimpleSpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SimpleSpeakerCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.flatten_dim = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_speakers)

    def _get_flatten_size(self):
        x = torch.zeros(1, 1, 128, 200)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def extract_embedding(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return x



# Model ResNet-18

class ResNetSpeaker(nn.Module):
    def __init__(self, num_speakers, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=num_speakers,
            in_chans=1
        )
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)
    
    def extract_embedding(self, x):
        embedding = self.base_model.forward_features(x)
        if embedding.ndim > 2:
            embedding = embedding.view(embedding.size(0), -1)
        return embedding


def get_model(model_type, num_speakers):
    if model_type == "baseline":
        return SpeakerCNN(num_speakers)
    elif model_type == "simple":
        return SimpleSpeakerCNN(num_speakers)
    elif model_type == "resnet":
        return ResNetSpeaker(num_speakers, pretrained=True)
    else:
        raise ValueError(f"Model necunoscut: {model_type}")

