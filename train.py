import glob

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import TripletFingerprintDataset
from network import TripletNet, EmbeddingNet
from losses import TripletLoss
from trainer import fit
from dataset import get_img_label

# Device
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Hyperparameters
in_channel = 1
batch_size = 128
learning_rate = 0.001
step_size = 50
num_epochs = 50


# Load Data
img_dir = sorted(glob.glob('train_data/*.jpg'))
classes = [int(i) for i in range(get_img_label(img_dir[-1]))]

transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

dataset = TripletFingerprintDataset(img_dir, classes, transform=transforms)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)

model.to(device)

# Loss and Optimizer
margin = 1.
loss = TripletLoss(margin)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

# Train Network
fit(train_loader, test_loader, model, loss, optimizer, scheduler, num_epochs, cuda)

# Save network
path = "model.pth"
torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)