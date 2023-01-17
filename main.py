import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
from torch.profiler import profile, record_function, ProfilerActivity

from simulating_random_sample import train_classifier_only

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
from utils import *
import arguments

import wandb
from dotenv import load_dotenv

load_dotenv()

wandb.login(key=os.environ['WANDB_KEY'])

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False,num_workers=args.num_workers)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    
    elif 'snapshot_serengeti' in args.dataset:
        animal_test_dataset = BoundingBoxImageLoader(
            # pickle_file=args.data_path+'/'+'df_metadata_test.df', # load test dataframe
            test_file_name = 'df_balanced_top_10_metadata_test.df' if args.dataset == 'snapshot_serengeti_balanced_top_10' else 'df_metadata_test.df'
            pickle_file=os.environ['DATA_DIR_PATH']+'/'+ test_file_name,
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=augmentations_medium())

        test_dataloader = data.DataLoader(animal_test_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, worker_init_fn=set_worker_sharing_strategy)

        if args.dataset == 'snapshot_serengeti_balanced_top_10':
            train_file_name = 'df_balanced_top_10_metadata_train.df'
        else:
            train_file_name = 'df_balanced_metadata_train.df' if args.balanced else 'df_metadata_train.df'
        train_dataset = BoundingBoxImageLoader(
            # pickle_file=args.data_path+'/'+'df_metadata_train.df', # load train dataframe
            pickle_file=os.environ['DATA_DIR_PATH']+'/'+train_file_name,
            root_dir=os.environ['DATA_DIR_PATH'],
            transform=augmentations_medium()
        )

        if args.dataset == 'snapshot_serengeti_complete':
            args.num_val = 127725
            args.num_images = 1277251
            args.budget = 63862
            args.initial_budget = 127725
            args.num_classes = 47
        elif args.dataset == 'snapshot_serengeti_random':
            args.num_val = 10000
            args.num_images = 1277251
            args.budget = 5000
            args.initial_budget = 10000
            args.num_classes = 47
        elif args.dataset == 'snapshot_serengeti_10k':
            args.num_val = 10000
            args.num_images = 1277251
            args.budget = 5000
            args.initial_budget = 10000
            args.num_classes = 47
        elif args.dataset == 'snapshot_serengeti_balanced_random':
            args.num_val = 10000
            args.num_images = 451452
            args.budget = 5000
            args.initial_budget = 10000
            args.num_classes = 47
        elif args.dataset == 'snapshot_serengeti_balanced_10k':
            args.num_val = 10000
            args.num_images = 451452
            args.budget = 5000
            args.initial_budget = 10000
            args.num_classes = 47
        elif args.dataset == 'snapshot_serengeti_balanced_top_10':
            args.num_val = 10000
            args.num_images = 175000
            args.budget = 5000
            args.initial_budget = 10000
            args.num_classes = 10
        else:
            args.num_val = 5000
            args.num_images = 50000
            args.budget = 2500
            args.initial_budget = 5000
            args.num_classes = 47
    
    else:
        raise NotImplementedError
        
    all_indices = set(np.arange(args.num_images))
    # fix starting conditions for random baseline and 10k
    # for more accurate comparison
    fixed_initial_sampler = random.Random()
    fixed_initial_sampler.seed(1234)
    val_indices = fixed_initial_sampler.sample(all_indices, args.num_val)
    # val_indices = random.sample(all_indices, args.num_val)

    all_indices = np.setdiff1d(list(all_indices), val_indices)
    
    initial_indices = fixed_initial_sampler.sample(list(all_indices), args.initial_budget)
    # initial_indices = random.sample(list(all_indices), args.initial_budget)

    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
            worker_init_fn=set_worker_sharing_strategy)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers,
            worker_init_fn=set_worker_sharing_strategy)
            
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    # args.cuda = False
    solver = Solver(args, test_dataloader,device)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    with wandb.init(project="vaal-log"):
        # wandb.log({'intitial_samples':current_indices})
        for split in splits:
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            task_model = vgg.vgg16_bn(num_classes=args.num_classes)
            vae = model.VAE(args.latent_dim)
            discriminator = model.Discriminator(args.latent_dim)

            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
            unlabeled_dataloader = data.DataLoader(train_dataset, 
                    sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False,
                    num_workers=args.num_workers)

            if 'random' in args.dataset:
                acc= train_classifier_only(args,querry_dataloader,val_dataloader,test_dataloader,task_model,unlabeled_dataloader)
                print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
                accuracies.append(acc)
                sampled_indices = random.sample(list(unlabeled_indices),args.budget)

            else:
                # train the models on the current data
                acc, vae, discriminator = solver.train(querry_dataloader,
                                                    val_dataloader,
                                                    task_model, 
                                                    vae, 
                                                    discriminator,
                                                    unlabeled_dataloader)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
                # prof.export_chrome_trace("trace.json")
                print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
                accuracies.append(acc)

                sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)

            current_indices = list(current_indices) + list(sampled_indices)
            sampler = data.sampler.SubsetRandomSampler(current_indices)
            querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                    batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                    worker_init_fn=set_worker_sharing_strategy)
    
    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

