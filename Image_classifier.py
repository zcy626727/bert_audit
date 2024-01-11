import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F

class Imageclassifier:
    def __init__(self) -> None:
        self.model = BinaryClassifier()
        self.model.load_state_dict(torch.load('./image_model/model.pth'))
        self.model.eval

    def upload(self,pic):
        face = cv2.resize(pic, (128, 128), interpolation=cv2.INTER_CUBIC)
        face_normalized=face.transpose((2, 0, 1))/255.0
        face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.Tensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(face_tensor)
        print(outputs.item())
        return outputs.item()<0.5
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Compute spatial attention weights
        x1 = self.conv1(x)
        x2 = self.conv2(F.relu(x1))
        weights = torch.sigmoid(self.conv3(F.relu(x2)))
        
        # Apply spatial attention to input features
        x = x * weights.expand_as(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.attention = AttentionModule(256, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

