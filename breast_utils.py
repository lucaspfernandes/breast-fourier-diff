import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule
import glob
from PIL import Image
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda')
img_size = 224
batch_size = 20

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    max_val = -np.inf
    min_val = np.inf
    for images in loader:
#        batch_size, num_channels, height, width = images.shape
#        num_pixels += batch_size * height * width
#        mean += images.mean(axis=(0, 2, 3)).sum()
#        std += images.std(axis=(0, 2, 3)).sum()
        img_min = images.min()
        img_max = images.max()
        if img_max > max_val:
            max_val = img_max
        if img_min < min_val:
            min_val = img_min
#    mean /= num_pixels
#    std /= num_pixels
#    return mean, std
    return max_val, min_val


class BreastDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_files = sorted(glob.glob(f'{root_dir}/*_CROP/*.tiff'))
        self.transform = transform  

    def __len__(self):
            return len(self.img_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.img_files[idx])
        img = img.convert('F')
        if self.transform:
            img = self.transform(img)

        return img

transform_list_norm = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=None)])
breast_norm_dataset = BreastDataset(root_dir='../data/MAMA', transform=transform_list_norm)
breast_norm_loader = DataLoader(breast_norm_dataset, batch_size=32, shuffle=True)
# breast_mean, breast_std = get_mean_std(breast_norm_loader)
breast_max, breast_min = get_mean_std(breast_norm_loader)


transform_list = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize((img_size, img_size), antialias=None),
                                    transforms.Normalize(mean=breast_min, std=(breast_max-breast_min)),
                                    transforms.Normalize(mean=0.5, std=0.5)])
breast_dataset = BreastDataset(root_dir='../data/MAMA', transform=transform_list)
breast_train_loader = DataLoader(breast_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.batchnorm1(self.conv1(x)))
        x = torch.nn.functional.relu(self.batchnorm2(self.conv2(x)))
        return x
    
class FullyConnectedBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)
        self.batchnorm1 = torch.nn.BatchNorm1d(out_channels)
        self.batchnorm2 = torch.nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.batchnorm1(self.fc1(x)))
        x = torch.nn.functional.relu(self.batchnorm2(self.fc2(x)))
        return x

class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_base_filters=32):
        super().__init__()

        self.image_size = img_size
        # encoder
        self.conv1 = ConvolutionalBlock(in_channels, num_base_filters) #32
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvolutionalBlock(num_base_filters, num_base_filters*2) #64
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvolutionalBlock(num_base_filters*2, num_base_filters*4) #128
        self.pool6 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = ConvolutionalBlock(num_base_filters*4, num_base_filters*8) #256
        self.pool8 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # bottleneck
        self.conv9 = ConvolutionalBlock(num_base_filters*8, num_base_filters*16) #512
#        self.flatten6 = torch.nn.Flatten()
#        self.fc7 = FullyConnectedBlock((self.image_size//4)*(self.image_size//4)*num_base_filters*4, 1024)
#        self.fc8 = FullyConnectedBlock(1024, (self.image_size//4)*(self.image_size//4)*num_base_filters*4)
#        self.unflatten9 = torch.nn.Unflatten(1, (num_base_filters*4, self.image_size//4, self.image_size//4))
        # decoder
        self.upconv10 = torch.nn.ConvTranspose2d(num_base_filters*16, num_base_filters*8, kernel_size=2, stride=2)
        self.conv11 = ConvolutionalBlock(num_base_filters*16, num_base_filters*8)
        self.upconv12 = torch.nn.ConvTranspose2d(num_base_filters*8, num_base_filters*4, kernel_size=2, stride=2)
        self.conv13 = ConvolutionalBlock(num_base_filters*8, num_base_filters*4)
        self.upconv14 = torch.nn.ConvTranspose2d(num_base_filters*4, num_base_filters*2, kernel_size=2, stride=2)
        self.conv15 = ConvolutionalBlock(num_base_filters*4, num_base_filters*2)
        self.upconv16 = torch.nn.ConvTranspose2d(num_base_filters*2, num_base_filters, kernel_size=2, stride=2)
        self.conv17 = ConvolutionalBlock(num_base_filters*2, num_base_filters)
        self.conv18 = torch.nn.Conv2d(num_base_filters, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x) #skip
        x2 = self.pool2(x1)
        x3 = self.conv3(x2) #skip 
        x4 = self.pool4(x3)
        x5 = self.conv5(x4) #skip
        x6 = self.pool6(x5)
        x7 = self.conv7(x6) #skip
        x8 = self.pool8(x7)
        x9 = self.conv9(x8)
  #      x6 = self.flatten6(x5)
  #      x7 = self.fc7(x6)
  #      x8 = self.fc8(x7)
  #      x9 = self.unflatten9(x8)
        x10 = self.upconv10(x9)
        x11 = self.conv11(torch.cat((x10, x7), dim=1))
        x12 = self.upconv12(x11)
        x13 = self.conv13(torch.cat((x12, x5), dim=1))
        x14 = self.upconv14(x13)
        x15 = self.conv15(torch.cat((x14, x3), dim=1))
        x16 = self.upconv16(x15)
        x17 = self.conv17(torch.cat((x16, x1), dim=1))
        x18 = self.conv18(x17)
        return x18


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels=32, expandToImage=True):
        super().__init__()
        self.fc1 = FullyConnectedBlock(1, 256)
        self.fc2 = FullyConnectedBlock(256, 256)
        self.fc3 = FullyConnectedBlock(256, out_channels)
        self.expandToImage = expandToImage
    def forward(self, t):
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        x = self.fc1(t)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.expandToImage:
            t_enc = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img_size, img_size)
        else:
            t_enc = x
        return t_enc

class PositionEncoder(torch.nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        
        xGrid, yGrid = torch.meshgrid(torch.linspace(-1, 1, img_size), torch.linspace(-1, 1, img_size))
        xGrid = xGrid.unsqueeze(0).unsqueeze(0)
        yGrid = yGrid.unsqueeze(0).unsqueeze(0)
        rGrid = torch.sqrt(xGrid**2 + yGrid**2)
        self.xyGrid = torch.cat((xGrid, yGrid,rGrid), dim=1).to(device)

        self.conv1 = ConvolutionalBlock(3, 32)
        self.conv2 = ConvolutionalBlock(32, 32)
        self.conv3 = ConvolutionalBlock(32, out_channels)
    
    def forward(self):
        x = self.conv1(self.xyGrid)
        x = self.conv2(x)
        pos_enc = self.conv3(x)
        return pos_enc
        

class BayesianClassifier(torch.nn.Module):
    def __init__(self, time_encoder, input_channels, output_channels):
        super().__init__()
        self.time_encoder = time_encoder
        self.conv1 = ConvolutionalBlock(input_channels, 32)
        self.conv2 = ConvolutionalBlock(32, 32)
        self.conv3 = ConvolutionalBlock(32, 32)
        self.fc1 = FullyConnectedBlock(32*img_size*img_size, 512)
        self.fc2 = FullyConnectedBlock(512, 256)
        self.fc3 = FullyConnectedBlock(256, output_channels)
        

    def forward(self, x_t, t):
        t_enc = self.time_encoder(t)
        x = torch.concat((x_t, t_enc), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 32*img_size*img_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x =  torch.nn.functional.softmax(x, dim=1)
        x = x + 1e-4  # To avoid log(0)
        x = x / x.sum(dim=1, keepdim=True)
        return x