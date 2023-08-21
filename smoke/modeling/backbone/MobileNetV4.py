import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the the input tensor shape is: torch.Size([2, 3, 384, 1280])
# the output tensor shape is: torch.Size([2, 64, 96, 320])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = EfficientNet.from_pretrained('efficientnet-b0')

    def forward(self, x):
        x = self.mobilenet(x)
        return x

model = Model().to(device)

input_tensor = torch.rand(2,3,384,1280).to(device)
output = model(input_tensor)

print(input_tensor.shape)
print(output.shape)