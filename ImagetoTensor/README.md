> Torch 에서 이미지를 사용하기 위해서는 numpy array 형태로 불러온 후 torch.Tensor로 변환해 주어야함

## 1. 이미지 파일을 텐서로 변환 및 정규화 
```python
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

trans = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
trainset = torchvision.datasets.ImageFolder(root = f'{DIR}', transform = trans)
```
