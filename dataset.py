import random
from PIL import Image

from torch.utils.data import Dataset


def get_img_label(img_fp):
    img_fn = img_fp.split('/')[-1]
    img_label = img_fn.split('_')[0]
    return int(img_label)


class TripletFingerprintDataset(Dataset):
    def __init__(self, imgs_fp, classes, transform=None):
        self.imgs_fp = imgs_fp
        self.classes = classes
        self.img_label_dict = {get_img_label(img_fp): [img_fp] for img_fp in self.imgs_fp}
        self.transform = transform

    def __getitem__(self, idx):
        def generate_triplets(idx):
            anchor_fp = self.imgs_fp[idx]
            anchor_class = get_img_label(anchor_fp)
            pos_fp = random.choice(self.img_label_dict[anchor_class])
            neg_class = random.randint(0, len(self.classes) - 1)
            while neg_class == anchor_class:
                neg_class = random.randint(0, len(self.classes) - 1)
            neg_fp = random.choice(self.img_label_dict[neg_class])
            anchor = Image.open(anchor_fp).convert('L')
            pos = Image.open(pos_fp).convert('L')
            neg = Image.open(neg_fp).convert('L')

            return anchor, pos, neg

        anchor, pos, neg = generate_triplets(idx)
        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return (anchor, pos, neg), []

    def __len__(self):
        return len(self.imgs_fp)
