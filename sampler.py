import torch

import numpy as np
import pandas as pd
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
class AdversarySampler:
    def __init__(self, budget, device, image_size):
        self.budget = budget
        self.device = device
        self.image_size = image_size


    def sample(self, vae, discriminator, unlabeled_data, cuda, split,task_model,labeled_data):
        all_preds = []
        all_indices = []
        tsne_embeddings = []
        task_predictions = []
        labels_list = []
        images_list = []
        tmp = []

        for images, labels, indices in unlabeled_data:
            if cuda:
                # images = images.cuda()
                images = images.to(self.device)

            with torch.no_grad():
                _, z, mu, _ = vae(images)
                preds = discriminator(mu)
                task_preds = task_model(images)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
            tsne_embeddings.append(mu.cpu().numpy())
            t_preds = torch.nn.functional.softmax(task_preds.cpu(),dim=1)
            entropy = -torch.sum(t_preds * torch.log2(t_preds),axis=1)
            task_predictions.extend(entropy)
            labels_list.append(labels)
            images_list.append(images.cpu().numpy())

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        tsne_embeddings = np.concatenate(tsne_embeddings, axis=0)
        task_predictions = torch.stack(task_predictions)
        task_predictions = task_predictions.view(-1)
        labels_list = np.concatenate(labels_list, axis=0)
        images_list = np.concatenate(images_list, axis=0).reshape(-1,3,self.image_size,self.image_size)

        #load labled images for visualisation
        labeled_all_indices = []
        labeled_tsne_embeddings = []
        labeled_labels_list = []
        labeled_images_list = []

        for images, labels, indices in labeled_data:
            if cuda:
                # images = images.cuda()
                images = images.to(self.device)

            with torch.no_grad():
                _, z, mu, _ = vae(images)

            labeled_all_indices.extend(indices)
            labeled_tsne_embeddings.append(mu.cpu().numpy())
            labeled_labels_list.append(labels)
            labeled_images_list.append(images.cpu().numpy())

        labeled_tsne_embeddings = np.concatenate(labeled_tsne_embeddings, axis=0)
        labeled_labels_list = np.concatenate(labeled_labels_list, axis=0)
        labeled_images_list = np.concatenate(labeled_images_list, axis=0).reshape(-1,3,self.image_size,self.image_size)

        #logging to wandb
        data = [[pred] for pred in all_preds.tolist()]
        table = wandb.Table(data=data, columns=["scores"])
        wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
        title="Prediction Score Distribution")})

        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]
        print(list(querry_pool_indices[:200]))

        _, not_querry_indices = torch.topk(all_preds, 200,largest=False)
        not_querry_pool_indices = np.asarray(all_indices)[not_querry_indices]
        print(list(not_querry_pool_indices[:200]))

        tsne = TSNE()
        new_embeddings = tsne.fit_transform(np.concatenate((tsne_embeddings, labeled_tsne_embeddings), axis=0))
        d = {'feature_1':new_embeddings[:len(tsne_embeddings),0], 'feature_2':new_embeddings[:len(tsne_embeddings),1], 'index':np.asarray(all_indices), 'labels':labels_list, 'images':list(images_list)}
        d = pd.DataFrame(data=d)
        d['is_informative'] = d['index'].isin(querry_pool_indices)
        d['disc_preds'] = all_preds
        d['task_model_preds'] = task_predictions
        d['is_training'] = False

        # d.to_pickle(f'df_sample_tsne_{split}.df')
        
        # take highest entropy as sampling
        _, querry_indices = torch.topk(task_predictions, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]
        print(list(querry_pool_indices[:200]))

        # combining Disc and Entropy for sampling
        num_samples = len(d)//2
        s1 = d.sort_values(by='task_model_preds',ascending=False).iloc[:num_samples]
        querry_pool_indices = s1.sort_values(by='disc_preds',ascending=False)[:self.budget]['index'].tolist()
        querry_pool_indices = np.array(querry_pool_indices)

        # plot coordinates with sampled as hue
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(data=d,x='feature_1',y='feature_2',hue='is_informative',ax=ax)
        wandb.log({"tsne_plot": wandb.Image(fig,caption='sampling')})

        # plot coordinates with class as hue
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(data=d,x='feature_1',y='feature_2',hue='labels',ax=ax)
        wandb.log({"tsne_plot": wandb.Image(fig,caption='sampling with labels')})

        # plot coordinategs with images
        tx, ty = d.feature_1, d.feature_2
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 3000
        height = 3000
        max_dim = 32
        full_image = Image.new('RGB', (width, height))
        for i in range(len(d)):
            image = d.iloc[i].images 
            image = np.transpose(image,(1,2,0))
            tile = Image.fromarray(np.uint8(image*255),'RGB')
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.LANCZOS)
            full_image.paste(tile, (int((width-max_dim) * tx[i]),
                            int((height-max_dim) * ty[i])))
        
        wandb.log({"tsne_plot": wandb.Image(full_image,caption='og images')})

        # plot coordinategs with sampled images
        labeled_d = {'feature_1':new_embeddings[len(tsne_embeddings):,0], 'feature_2':new_embeddings[len(tsne_embeddings):,1], 'index':np.asarray(labeled_all_indices), 'labels':labeled_labels_list, 'images':list(labeled_images_list)}
        labeled_d = pd.DataFrame(data=labeled_d)
        labeled_d['is_informative'] = False
        labeled_d['is_training'] = True

        chosen_d = d[d['is_informative']==True]
        chosen_d.reset_index(inplace=True, drop=True)
        chosen_d = pd.concat([chosen_d, labeled_d], ignore_index=True)
        tx, ty = chosen_d.feature_1, chosen_d.feature_2
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 3000
        height = 3000
        max_dim = 32
        full_image = Image.new('RGB', (width, height))
        for i in range(len(chosen_d)):
            image = chosen_d.iloc[i].images 
            image = np.transpose(image,(1,2,0))
            tile = Image.fromarray(np.uint8(image*255),'RGB')
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.LANCZOS)
            if chosen_d.iloc[i].is_training:
                tile = ImageOps.expand(tile,border=5,fill='yellow')
            full_image.paste(tile, (int((width-max_dim) * tx[i]),
                            int((height-max_dim) * ty[i])))
        
        wandb.log({"tsne_plot": wandb.Image(full_image,caption='sampled images')})

        return querry_pool_indices
        
