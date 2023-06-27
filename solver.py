import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE

import sampler
import copy

import wandb
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Solver:
    def __init__(self, args, test_dataloader, device,image_size):
        self.args = args
        self.test_dataloader = test_dataloader
        self.device = device
        self.image_size = image_size

        self.bce_loss = nn.BCELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget,self.device,self.image_size)


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
            # vae = vae.cuda()
            # discriminator = discriminator.cuda()
            # task_model = task_model.cuda()
            vae = vae.to(self.device)
            discriminator = discriminator.to(self.device)
            task_model = task_model.to(self.device)
        
        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count != 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10
            load_img_start = time.time()
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            load_img_end = time.time()

            if self.args.cuda:
                # labeled_imgs = labeled_imgs.cuda()
                # unlabeled_imgs = unlabeled_imgs.cuda()
                # labels = labels.cuda()

                labeled_imgs = labeled_imgs.to(self.device)
                unlabeled_imgs = unlabeled_imgs.to(self.device)
                labels = labels.to(self.device)

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()


            # VAE step
            vae_start = time.time()
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                # print(f'recon: {recon.shape}')
                # print(f'z: {z.shape}')
                # print(f'mu: {mu.shape}')
                # print(f'logvar: {logvar.shape}')
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta, iter_count)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta, iter_count)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
                if self.args.cuda:
                    # lab_real_preds = lab_real_preds.cuda()
                    # unlab_real_preds = unlab_real_preds.cuda()
                    lab_real_preds = lab_real_preds.to(self.device)
                    unlab_real_preds = unlab_real_preds.to(self.device)

                               
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
                        # labeled_imgs = labeled_imgs.cuda()
                        # unlabeled_imgs = unlabeled_imgs.cuda()
                        # labels = labels.cuda()
                        labeled_imgs = labeled_imgs.to(self.device)
                        unlabeled_imgs = unlabeled_imgs.to(self.device)
                        labels = labels.to(self.device)
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
                    # lab_real_preds = lab_real_preds.cuda()
                    # unlab_fake_preds = unlab_fake_preds.cuda()
                    lab_real_preds = lab_real_preds.to(self.device)
                    unlab_fake_preds = unlab_fake_preds.to(self.device)
                
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
                        # labeled_imgs = labeled_imgs.cuda()
                        # unlabeled_imgs = unlabeled_imgs.cuda()
                        # labels = labels.cuda()
                        labeled_imgs = labeled_imgs.to(self.device)
                        unlabeled_imgs = unlabeled_imgs.to(self.device)
                        labels = labels.to(self.device)

            disc_end = time.time()
            if iter_count % 100 == 0:
                labeled_preds = labeled_preds.cpu()
                lab_real_preds = lab_real_preds.cpu()
                unlabeled_preds = unlabeled_preds.cpu()
                unlab_fake_preds = unlab_fake_preds.cpu()

                true_lab = torch.cat((lab_real_preds,unlab_fake_preds),0).detach()
                pred_lab = torch.cat((labeled_preds, unlabeled_preds),0).detach()
                true_lab_logging = true_lab
                pred_lab_logging = pred_lab
                true_lab = true_lab.round()
                pred_lab = pred_lab.round()

                #log classification of Discriminator
                dsc_acc = accuracy_score(y_true=true_lab, y_pred=pred_lab)
                dsc_precision = precision_score(y_true=true_lab, y_pred=pred_lab)
                dsc_recall = recall_score(y_true=true_lab, y_pred=pred_lab)
                dsc_f1 = f1_score(y_true=true_lab, y_pred=pred_lab)

                wandb.log({
                    'iteration':iter_count,
                    'task_loss':task_loss.item(),
                    'total_vae_loss':total_vae_loss.item(),
                    'total_discriminator_loss':dsc_loss.item(),
                    'vae_labeled_loss':unsup_loss.item(),
                    'vae_unlabeled_loss':transductive_loss.item(),
                    'adv_loss':adv_loss.item(),
                    'disc_labeled_loss':dsc_lab_loss.item(),
                    'disc_unlabeled_loss':dsc_unlab_loss.item(),
                    'disc_acc':dsc_acc,
                    'disc_precision':dsc_precision,
                    'disc_recall':dsc_recall,
                    'disc_f1':dsc_f1
                })
                
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

            if iter_count % 300 == 0:
                acc = self.validate(task_model, val_dataloader) # evaluate taskmodel on validation-set
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)
                wandb.log({'task_acc':acc})
                print('current step: {} acc: {}'.format(iter_count, acc))
                print('best acc: ', best_acc)

                #evaluate discriminator on validation-set
                acc_eval_disc = self.validate_discriminator(discriminator, vae, val_dataloader)
                wandb.log({'eval_disc_acc':acc_eval_disc})

                #logging histograms
                true_data_logging = [[pred] for pred in true_lab_logging.tolist()]
                true_table = wandb.Table(data=true_data_logging, columns=["scores"])

                pred_data_logging = [[pred] for pred in pred_lab_logging.tolist()]
                pred_table = wandb.Table(data=pred_data_logging, columns=["scores"])
                
                #logging confusion matrix
                cf = confusion_matrix(true_lab_logging,np.round(pred_lab_logging),labels=[0,1])
                df_cm = pd.DataFrame(cf, index = ['unlabeled','labeled'],
                columns = ['unlabeled','labeled'])
                fig, ax = plt.subplots(figsize=(8,8))
                sns.heatmap(df_cm,annot=True,ax=ax)
                wandb.log({"confusion_matrix": wandb.Image(fig,caption='train')})

                #logging tsne()
                tsne = TSNE()
                tsne_embeddings = np.concatenate([mu.detach().cpu(),unlab_mu.detach().cpu()],axis=0)
                new_embeddings = tsne.fit_transform(tsne_embeddings)
                d = {'feature_1':new_embeddings[:,0], 'feature_2':new_embeddings[:,1], 'index':np.arange(len(tsne_embeddings))}
                d = pd.DataFrame(data=d)
                d['is_informative'] = d['index'] < 64

                # d.to_pickle(f'df_train_tsne.df')

                fig, ax = plt.subplots(figsize=(6,6))
                sns.scatterplot(data=d,x='feature_1',y='feature_2',hue='is_informative',ax=ax,s=10)
                wandb.log({"tsne_plot": wandb.Image(fig,caption='train')})

                #log reconstruction of test images
                for test_imgs, labels in self.test_dataloader:
                    if self.args.cuda:
                        # test_imgs = test_imgs.cuda()
                        test_imgs = test_imgs.to(self.device)
                    break

                #log Encoder Decoder
                with torch.no_grad():
                    recon, _, _, _ = vae(labeled_imgs)
                    test_recon, _, _, _ = vae(test_imgs)

                img_true = wandb.Image(labeled_imgs, caption='True images')
                img_recon_train = wandb.Image(recon, caption="after encode-decoding")
                img_recon_test = wandb.Image(test_recon, caption='reconstruction test')
                wandb.log({
                    "True images": img_true,
                    'reconstructed images':img_recon_train,
                    'reconstructed test image': img_recon_test,
                    'y_true_hist': wandb.plot.histogram(true_table, "scores",title="True Score Distribution"),    
                    'y_pred_hist': wandb.plot.histogram(pred_table, "scores",title="Prediction Score Distribution")
                })


        if self.args.cuda:
            # best_model = best_model.cuda()
            best_model = best_model.to(self.device)

        final_accuracy = self.test(best_model,vae)
        wandb.log({'final_acc':final_accuracy})
        return final_accuracy, vae, discriminator,task_model


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, split,task_model,labeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda, split,task_model,labeled_dataloader)

        return querry_indices
                

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        Y_PREDS, Y_TRUE = [], []
        for imgs, labels, _ in loader:
            if self.args.cuda:
                # imgs = imgs.cuda()
                imgs = imgs.to(self.device)

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
            Y_PREDS.append(preds)
            Y_TRUE.append(labels)


        #logging confusion matrix
        Y_PREDS = np.concatenate(Y_PREDS, axis=0)
        Y_TRUE = np.concatenate(Y_TRUE, axis=0)
        cf = confusion_matrix(Y_TRUE, Y_PREDS, labels=np.arange(10))
        df_cm = pd.DataFrame(cf, index = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
        columns = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
        fig, ax = plt.subplots(figsize=(8,8))
        sns.heatmap(df_cm,annot=True,ax=ax)
        wandb.log({"confusion_matrix_validate": wandb.Image(fig,caption='validation')})
        
        return correct / total * 100

    def validate_discriminator(self,discriminator,vae,loader):
        vae.eval()
        discriminator.eval()
        Y_PREDS, Y_TRUE = [], []
        for imgs, lables, _ in loader:
            if self.args.cuda:
                imgs = imgs.to(self.device)

            with torch.no_grad():
                _, z, mu, _ = vae(imgs)
                preds = discriminator(mu)

            Y_PREDS.append(preds.cpu().numpy())

        Y_PREDS = np.concatenate(Y_PREDS, axis=0)
        Y_TRUE = np.zeros_like(Y_PREDS)
        cf = confusion_matrix(Y_TRUE, np.round(Y_PREDS), labels=np.arange(2))
        df_cm = pd.DataFrame(cf, index = ['unlabeled','labeled'],
        columns = ['unlabeled','labeled'])
        fig, ax = plt.subplots(figsize=(8,8))
        sns.heatmap(df_cm,annot=True,ax=ax)

        pred_data_logging = Y_PREDS
        pred_table = wandb.Table(data=pred_data_logging, columns=["scores"])
        wandb.log({
            'discriminator_evaluation_acc':(1 - np.round(Y_PREDS).mean()),
            "confusion_matrix_validate": wandb.Image(fig,caption='disc_validation'),
            'discriminator evaluation': wandb.plot.histogram(pred_table, "scores",title="disc Prediction Distribution")
        })
    def test(self, task_model,vae):
        task_model.eval()
        vae.eval()
        total, correct = 0, 0
        test_loss = []
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                # imgs = imgs.cuda()
                imgs = imgs.to(self.device)

            with torch.no_grad():
                preds = task_model(imgs)
                recon,_,mu,logvar = vae(imgs)
            
            loss = self.vae_loss(imgs, recon, mu, logvar, self.args.beta, 1)
            test_loss.append(loss.cpu().numpy())
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        
        test_loss = np.array(test_loss).mean()
        wandb.log({
            "test_loss":test_loss
        })
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta, iteration):
        # MSE = self.mse_loss(recon, x)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        MSE = nn.BCELoss(size_average=False)(recon, x) / x.size(0)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = KLD * beta
        if iteration % 100 == 0:
            wandb.log({
                        "MSE_reconstruction": MSE,
                        'Kull_Leibler_Divergence':KLD    
                    })
        return MSE + KLD
