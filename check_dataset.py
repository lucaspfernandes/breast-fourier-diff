from torchvision import transforms
from breast_utils import UNet, TimeEncoder, PositionEncoder, BreastDataset 
import torch
from mnist_utils import mnist_train 

transform_list = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
breast_train = BreastDataset(root_dir='./data/MAMA', transform=transform_list)

a = breast_train.__getitem__(0)
b = mnist_train.__getitem__(0)
print(a)
print(b)

