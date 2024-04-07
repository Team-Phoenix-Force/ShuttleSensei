from torch import nn
import torch

class ConvRally(nn.Module):
    def __init__(self, in_frames):
        super().__init__()
        # Define video to image transformation layers
        self.vid_to_img = nn.Sequential(
            # 3D convolutional layer with batch normalization
            nn.Conv3d(in_channels=in_frames, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),  # Applies GELU activation function
            nn.MaxPool3d(kernel_size=3,stride=1, padding=(0,1,1)),  # 3D max pooling layer
            nn.Flatten(start_dim=2, end_dim=3),  # Flatten the output for transition to 2D convolution
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Adaptive average pooling layer
        self.avg = nn.AdaptiveAvgPool2d((8,8))
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(4096 * 8 * 8, 1024),
            nn.GELU(),
            nn.Dropout(),  # Dropout layer for regularization
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)  # Softmax activation for classification
        )

    def forward(self, x):
        # Forward pass through video to image transformation layers
        x = self.vid_to_img(x)
        # Adaptive average pooling
        x = self.avg(x)
        x = torch.flatten(x, 1)
        # Forward pass through classifier layers
        x = self.classifier(x)
        x = x[:, 0]  # Output only the first column
        return x
