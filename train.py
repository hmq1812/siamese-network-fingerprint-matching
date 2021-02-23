import glob

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import FingerprintDataset, BalancedBatchSampler, get_img_label
from triplet_selector import HardestNegativeTripletSelector, SemihardNegativeTripletSelector, RandomNegativeTripletSelector
from network import EmbeddingNet
from losses import OnlineTripletLoss
from train_module import fit

# Device
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Hyperparameters
in_channel = 1
batch_size = 128
learning_rate = 0.001
step_size = 50
num_epochs = 200


# Load Data
train_dir = sorted(glob.glob('train_data/train_set/*.jpg'))
valid_dir = sorted(glob.glob('train_data/valid_set/*.jpg'))

train_label = [get_img_label(img) for img in train_dir]
valid_label = [get_img_label(img) for img in valid_dir]

transforms = transforms.Compose([
    transforms.ToTensor()
])
train_set = FingerprintDataset(train_dir, transforms)
valid_set = FingerprintDataset(valid_dir, transforms)

train_batch_sampler = BalancedBatchSampler(train_label, n_classes=50, n_samples=4)
test_batch_sampler = BalancedBatchSampler(valid_label, n_classes=50, n_samples=4)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = DataLoader(dataset=train_set, batch_sampler=train_batch_sampler, **kwargs)
valid_loader = DataLoader(dataset=valid_set, batch_sampler=test_batch_sampler, **kwargs)

# Model
model = EmbeddingNet()

model.to(device)

# Loss and Optimizer
margin = 1.
loss = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

# Train Network
fit(train_loader, valid_loader, model, loss, optimizer, scheduler, num_epochs, cuda)
