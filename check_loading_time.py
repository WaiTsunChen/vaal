from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, data
import torchvision
from torchvision.io import read_image
import torch

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os 
import random
import time



class BoundingBoxImageLoader (Dataset):
    """Animal Bounding Box Crop."""

    def __init__(self, pickle_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_pickle(pickle_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 0]+'.JPG')
        image = read_image(img_name)
        bbox_im = self.dataframe.iloc[idx, 1]
        #image_croped = image.crop((bbox_im[0], bbox_im[1], bbox_im[0]+bbox_im[2], bbox_im[1]+bbox_im[3]))
        image_croped = transforms.functional.crop(
            image, int(bbox_im[1]), int(bbox_im[0]), int(bbox_im[3]), int(bbox_im[2])) # top, left, height, width
        
        sample = image_croped
        # sample = image
        target = self.dataframe.iloc[idx, 2]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, target, idx 


def read_data(dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

def augmentations_light():
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop((96, 96)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def augmentations_medium():
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop((96, 96)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    ]
)

num_val = 500
num_images = 5000  
budget = 250
initial_budget = 500
num_classes = 47
batch_size = 128

animal_train_dataset = BoundingBoxImageLoader(
    pickle_file='df_metadata_train.df', 
    root_dir=os.environ['DATA_DIR_PATH'],
    transform=augmentations_medium())

animal_test_dataset = BoundingBoxImageLoader(
    pickle_file='df_metadata_test.df', # load test dataframe
    root_dir=os.environ['DATA_DIR_PATH'],
    transform=augmentations_medium())

test_dataloader = data.DataLoader(animal_test_dataset, batch_size=batch_size, shuffle=True)


all_indices = set(np.arange(num_images))
val_indices = random.sample(all_indices, num_val)
all_indices = np.setdiff1d(list(all_indices), val_indices)

initial_indices = random.sample(list(all_indices), initial_budget)
sampler = data.sampler.SubsetRandomSampler(initial_indices)
val_sampler = data.sampler.SubsetRandomSampler(val_indices)

# dataset with labels available
querry_dataloader = data.DataLoader(animal_train_dataset, sampler=sampler, 
        batch_size=batch_size, drop_last=True, num_workers=0)
val_dataloader = data.DataLoader(animal_train_dataset, sampler=val_sampler,
        batch_size=batch_size, drop_last=False)

start = time.time()
labeled_data = read_data(querry_dataloader) 
check_one = time.time()
# unlabeled_data = read_data(unlabeled_dataloader, labels=False)
labeled_imgs, labels = next(labeled_data)
end = time.time()
print(f'total time: {end - start}')
print(f'loading querry data:{check_one-start}' )
print(f'get next batch time: {end-check_one}')
