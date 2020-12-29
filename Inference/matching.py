import numpy as np 
from PIL import Image
import torch
import torchvision.transforms as transforms

from network import EmbeddingNet, TripletNet
from config import *

class FingerPrintMatching(object):
    def __init__(self):
        self.threshold = 4
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.model = TripletNet(EmbeddingNet())
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        # self.model.to(device)

    def get_embedding(self, img):
        '''
        :img:   PIL object
        '''
        img = img.convert('L')
        img = img.resize((128, 128), Image.ANTIALIAS)
        transform = transforms.ToTensor()
        input = transform(img)
        input = input.unsqueeze(1).to(self.device)

        embedding = self.model.get_embedding(input)

        return embedding

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def match(self, x1, x2):
        dist = self.calc_euclidean(x1, x2)
        return 1 if dist < self.threshold else 0
