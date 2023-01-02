import torch

import numpy as np
import wandb
class AdversarySampler:
    def __init__(self, budget, device):
        self.budget = budget
        self.device = device


    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()
                # images = images.to(self.device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]
        print(querry_indices[:100])

        _, not_querry_indices = torch.topk(all_preds, int(self.budget),largest=False)
        not_querry_indices = np.asarray(all_indices)[not_querry_indices]
        print(not_querry_indices[:100])

        return querry_pool_indices
        
