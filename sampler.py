import torch

import numpy as np
import pandas as pd
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
class AdversarySampler:
    def __init__(self, budget, device):
        self.budget = budget
        self.device = device


    def sample(self, vae, discriminator, data, cuda, split):
        all_preds = []
        all_indices = []
        tsne_embeddings = []

        for images, _, indices in data:
            if cuda:
                # images = images.cuda()
                images = images.to(self.device)

            with torch.no_grad():
                _, z, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
            tsne_embeddings.append(z.cpu().numpy())

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        tsne_embeddings = np.concatenate(tsne_embeddings, axis=0)

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
        new_embeddings = tsne.fit_transform(tsne_embeddings)
        d = {'feature_1':new_embeddings[:,0], 'feature_2':new_embeddings[:,1], 'index':np.asarray(all_indices)}
        d = pd.DataFrame(data=d)
        d['is_informative'] = d['index'].isin(querry_pool_indices)

        # d.to_pickle(f'df_sample_tsne_{split}.df')

        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(data=d,x='feature_1',y='feature_2',hue='is_informative',ax=ax)
        wandb.log({"tsne_plot": wandb.Image(fig,caption='sampling')})

        return querry_pool_indices
        
