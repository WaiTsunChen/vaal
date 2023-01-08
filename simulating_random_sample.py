import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy

import wandb
import time

def read_data( dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def train_classifier_only(args,querry_dataloader, val_dataloader,test_dataloader, task_model, unlabeled_dataloader):
        args.train_iterations = (len(querry_dataloader.sampler) * args.train_epochs) // args.batch_size
        lr_change = args.train_iterations // 4
        labeled_data = read_data(querry_dataloader) 
        unlabeled_data = read_data(unlabeled_dataloader, labels=False)

        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        ce_loss = nn.CrossEntropyLoss()
        wandb.watch([task_model],log='all',log_freq=9000)

        task_model.train()

        if  args.cuda:
            task_model = task_model.cuda()
        
        best_acc = 0
        for iter_count in range( args.train_iterations):
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)


            if  args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 100 == 0:
                wandb.log({'iteration':iter_count})
                wandb.log({'task_loss':task_loss.item()})
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))

            if iter_count % 300 == 0:
                acc = validate(args,task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)
                wandb.log({'task_acc':acc})
                print('current step: {} acc: {}'.format(iter_count, acc))
                print('best acc: ', best_acc)


        if args.cuda:
            best_model = best_model.cuda()
            # best_model = best_model.to(self.device)

        final_accuracy = test(best_model,test_dataloader,task_model)
        return final_accuracy

def validate(args,task_model, loader):
    task_model.eval()
    total, correct = 0, 0
    for imgs, labels, _ in loader:
        if args.cuda:
            imgs = imgs.cuda()
            # imgs = imgs.to(self.device)

        with torch.no_grad():
            preds = task_model(imgs)

        preds = torch.argmax(preds, dim=1).cpu().numpy()
        correct += accuracy_score(labels, preds, normalize=False)
        total += imgs.size(0)
    return correct / total * 100

def test(args,test_dataloader, task_model):
    task_model.eval()
    total, correct = 0, 0
    for imgs, labels, _ in test_dataloader:
        if args.cuda:
            imgs = imgs.cuda()

        with torch.no_grad():
            preds = task_model(imgs)

        preds = torch.argmax(preds, dim=1).cpu().numpy()
        correct += accuracy_score(labels, preds, normalize=False)
        total += imgs.size(0)
    return correct / total * 100