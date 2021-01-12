import numpy as np 
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.spatial import distance

from network import EmbeddingNet
from config import *

class FingerPrintMatching(object):
    def __init__(self):
        self.threshold = 1.75
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.model = EmbeddingNet()
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        # self.model.to(device)

    def get_embedding(self, img):
        '''
        :img:   PIL object
        return: Embedding Tensor
        '''
        img = img.convert('L')
        img = img.resize((128, 128), Image.ANTIALIAS)
        transform = transforms.ToTensor()
        input = transform(img)
        input = input.unsqueeze(1).to(self.device)

        embedding = self.model.get_embedding(input)

        return embedding

    def calc_euclidean(self, x1, x2):
        # return ((x1 - x2).pow(2).sum(1)).pow(1/2)
        x1_np = x1.cpu().detach().numpy()
        x2_np = x2.cpu().detach().numpy()
        return distance.euclidean(x1_np, x2_np)

    def match(self, x1, x2):
        dist = self.calc_euclidean(x1, x2)
        return 1 if dist < self.threshold else 0

