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


class Solver:
    def __init__(self, args, test_dataloader, device):
        self.args = args
        self.test_dataloader = test_dataloader
        self.device = device

        self.bce_loss = nn.BCELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget,self.device)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img


    def train(self, querry_dataloader, val_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
        self.args.train_iterations = (len(querry_dataloader.sampler)* self.args.train_epochs) // self.args.batch_size
        lr_change = self.args.train_iterations // 4
        labeled_data = self.read_data(querry_dataloader) 
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        wandb.watch([vae,discriminator,task_model],log='all',log_freq=9000)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
            # vae = vae.to(self.device)
            # discriminator = discriminator.to(self.device)
            # task_model = task_model.to(self.device)
        
        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10
            load_img_start = time.time()
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            load_img_end = time.time()

            if self.args.cuda:
                img_to_gpu_start = time.time()
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()
                img_to_gpu_end = time.time()
                # labeled_imgs = labeled_imgs.to(self.device)
                # unlabeled_imgs = unlabeled_imgs.to(self.device)
                # labels = labels.to(self.device)

            # task_model step
            task_start =time.time()
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()
            task_end = time.time()

            # VAE step
            vae_start = time.time()
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                # print(f'recon: {recon.shape}')
                # print(f'z: {z.shape}')
                # print(f'mu: {mu.shape}')
                # print(f'logvar: {logvar.shape}')
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()
                    # lab_real_preds = lab_real_preds.to(self.device)
                    # unlab_real_preds = unlab_real_preds.to(self.device)

                               
                adv_loss = self.bce_loss(labeled_preds, lab_real_preds.unsqueeze(1)) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds.unsqueeze(1))
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * adv_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
                        # labeled_imgs = labeled_imgs.to(self.device)
                        # unlabeled_imgs = unlabeled_imgs.to(self.device)
                        # labels = labels.to(self.device)
            vae_end = time.time()
            disc_start =time.time()
            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                    # lab_real_preds = lab_real_preds.to(self.device)
                    # unlab_fake_preds = unlab_fake_preds.to(self.device)
                
                dsc_lab_loss = self.bce_loss(labeled_preds, lab_real_preds.unsqueeze(1))
                dsc_unlab_loss = self.bce_loss(unlabeled_preds, unlab_fake_preds.unsqueeze(1))
                dsc_loss =  dsc_lab_loss +dsc_unlab_loss

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()
                        # labeled_imgs = labeled_imgs.to(self.device)
                        # unlabeled_imgs = unlabeled_imgs.to(self.device)
                        # labels = labels.to(self.device)

            disc_end = time.time()
            # wandb.log({'get next images':load_img_end-load_img_start})
            # wandb.log({'images to GPU':img_to_gpu_end-img_to_gpu_start})
            # wandb.log({'task model':task_end-task_start})
            # wandb.log({'vae part':vae_end-vae_start})
            # wandb.log({'disc part':disc_end-disc_start})
            if iter_count % 100 == 0:
                wandb.log({'iteration':iter_count})
                wandb.log({'task_loss':task_loss.item()})
                wandb.log({'total_vae_loss':total_vae_loss.item()})
                wandb.log({'total_discriminator_loss':dsc_loss.item()})

                wandb.log({'vae_labeled_loss':unsup_loss.item()})
                wandb.log({'vae_unlabeled_loss':transductive_loss.item()})
                wandb.log({'adv_loss':adv_loss.item()})

                wandb.log({'disc_labeled_loss':dsc_lab_loss.item()})
                wandb.log({'disc_unlabeled_loss':dsc_unlab_loss.item()})

                wandb.log({"dsc_conf_mat_labeled" : wandb.plot.confusion_matrix(probs=None,
                                        y_true= labeled_preds, preds=lab_real_preds.unsqueeze(1),
                                        )})
                wandb.log({"dsc_conf_mat_unlabeled" : wandb.plot.confusion_matrix(probs=None,
                                        y_true= labeled_preds, preds=lab_real_preds.unsqueeze(1),
                                        )})


                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

            if iter_count % 300 == 0:
                acc = self.validate(task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)
                wandb.log({'task_acc':acc})
                print('current step: {} acc: {}'.format(iter_count, acc))
                print('best acc: ', best_acc)


        if self.args.cuda:
            best_model = best_model.cuda()
            # best_model = best_model.to(self.device)

        final_accuracy = self.test(best_model)
        wandb.log({'final_acc':final_accuracy})
        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices
                

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()
                # imgs = imgs.to(self.device)

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()
                # imgs = imgs.to(self.device)

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
