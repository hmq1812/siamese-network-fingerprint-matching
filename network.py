import torch.nn as nn


class EmbeddingNet(nn.Module):
    # Input size = (1, 128, 128)
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(1,1)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=(1,1)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1,1)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                     )

        self.fc = nn.Sequential(nn.Linear(128 * 16 * 16, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, pos, neg):
        anchor_embedding = self.embedding_net(anchor)
        pos_embedding = self.embedding_net(pos)
        neg_embedding = self.embedding_net(neg)
        return anchor_embedding, pos_embedding, neg_embedding

    def get_embedding(self, x):
        return self.embedding_net(x)
