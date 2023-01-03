from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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
from custom_datasets import *
from dotenv import load_dotenv

load_dotenv()

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

num_val = 127725
num_images = 1277251
budget = 63862
initial_budget = 127725
num_classes = 47
batch_size = 128

animal_train_dataset = BoundingBoxImageLoader(
    pickle_file=os.environ['DATA_DIR_PATH']+'/'+'df_metadata_train.df', 
    root_dir=os.environ['DATA_DIR_PATH'],
    transform=augmentations_medium())

animal_test_dataset = BoundingBoxImageLoader(
    pickle_file=os.environ['DATA_DIR_PATH']+'/'+'df_metadata_test.df',  # load test dataframe
    root_dir=os.environ['DATA_DIR_PATH'],
    transform=augmentations_medium())

test_dataloader = DataLoader(animal_test_dataset, batch_size=batch_size, shuffle=True,num_workers=32)
train_dataloader = DataLoader(animal_test_dataset,batch_size=batch_size, shuffle=True,num_workers=32)

count_train = 0
for (sample, target,idx) in train_dataloader:
    count_train += 1
count_validation = 0
for (sample, target,idx) in test_dataloader:
    count_validation +=1
print(f'count_train: {count_train}')
print(f'count_validation: {count_validation}')

# all_indices = set(np.arange(num_images))
# val_indices = random.sample(all_indices, num_val)
# all_indices = np.setdiff1d(list(all_indices), val_indices)

# initial_indices = random.sample(list(all_indices), initial_budget)
# sampler = data.sampler.SubsetRandomSampler(initial_indices)
# val_sampler = data.sampler.SubsetRandomSampler(val_indices)

# # dataset with labels available
# querry_dataloader = data.DataLoader(animal_train_dataset, sampler=sampler, 
#         batch_size=batch_size, drop_last=True, num_workers=0)
# val_dataloader = data.DataLoader(animal_train_dataset, sampler=val_sampler,
#         batch_size=batch_size, drop_last=False)

# unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
# unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
# unlabeled_dataloader = data.DataLoader(train_dataset, 
#                         sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False,
#                     num_workers=args.num_workers)

# labeled_data = read_data(querry_dataloader) 
# unlabeled_data = read_data(unlabeled_dataloader, labels=False)
