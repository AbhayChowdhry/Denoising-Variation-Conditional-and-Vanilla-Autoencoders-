from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
import torchvision.models.mobilenet as mobilenet
import torchvision.models.mobilenetv2 as mobilenetv2
import torchvision.datasets.folder as folder
import torchvision.models as models
from torchvision.datasets.folder import default_loader
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import itertools

EPOCH = 50

aug_images = os.listdir(aug_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlteredMNIST(Dataset):
    def __init__(self, transform=None):
        self.aug_path = aug_path
        self.clean_path = clean_path
        self.transform = transform
        self.aug_images = os.listdir(aug_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.aug_images)

    def __getitem__(self, idx):
        max_clean_img_path = mappings[idx]
        aug_filename = self.aug_images[idx]
        aug_img_path = os.path.join(self.aug_path, aug_filename) 

        aug_img = folder.default_loader(aug_img_path)
        clean_img = folder.default_loader(max_clean_img_path)

        aug_img = self.transforms(aug_img)
        clean_img = self.transforms(clean_img)

        return aug_img, clean_img
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64), 
            ResNetBlock(64, 64)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_mu = nn.Linear(64*14*14, 64)
        self.fc_logvar = nn.Linear(64*14*14, 64)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(64, 64 * 14 * 14)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.conv_transpose2 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, x):
        if isinstance(x, tuple):
            x, logvar = x
        out = self.fc(x)
        out = out.view(-1, 64, 14, 14)
        out = self.conv_transpose1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        out = self.conv_transpose2(out)
        out = self.bn2(out)
        out = torch.sigmoid(out)
        return out

import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, 0, 0


class VariationalDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalDenoisingAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar


def ParameterSelector(E, D):
    parameters_to_train = list(E.parameters()) + list(D.parameters())
    return parameters_to_train


class AELossFn:
    def calculate_loss(self, recon_batch, clean_images):
        return F.mse_loss(recon_batch, clean_images)

class VAELossFn:
    def calculate_loss(self, recon_batch, clean_images, mu, logvar):
        MSE = F.mse_loss(recon_batch, clean_images, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

def plot_tsne_embeddings(encoder, dataloader, device, model_type, epoch):
    all_embeddings = []
    all_labels = []

    # Compute embeddings for the entire dataset
    with torch.no_grad():
        for aug_images, _ in dataloader:
            aug_images = aug_images.to(device)
            mu, _ = encoder(aug_images)
            all_embeddings.append(mu.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings)

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=10, n_iter=250)
    tsne_embeddings = tsne.fit_transform(all_embeddings)

    # Plot 3D t-SNE embeddings
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], marker='o')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D t-SNE Embedding Plot')
    plt.show()

    # Save the plot
    plt.savefig(f'{model_type}_epoch_{epoch}.png')

def calculate_ssim(model, test_dataloader, device, name):
    model.eval()  # Set model to evaluation mode
    ssim_scores = []

    num_samples = int(len(test_dataloader.dataset) * 0.2)
    test_dataloader = itertools.islice(test_dataloader, num_samples)

    idx = 1
    with torch.no_grad():
        for aug_images, clean_images in test_dataloader:
            print(idx, end='\r')
            aug_images, clean_images = aug_images.to(device), clean_images.to(device)

            recon_images, _, _ = model(aug_images)

            # Convert images to numpy arrays and move to CPU
            recon_images = recon_images.cpu().numpy().transpose(0, 2, 3, 1)
            clean_images = clean_images.cpu().numpy().transpose(0, 2, 3, 1)

            # Calculate SSIM score for each pair of images in the batch
            for recon_img, clean_img in zip(recon_images, clean_images):
                score = ssim(recon_img, clean_img, channel_axis=2, data_range=1.0)  # Use channel_axis instead of multichannel
                ssim_scores.append(score)

    # Calculate average SSIM score
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    return avg_ssim

class AETrainer:
    def __init__(self, Data, E, D, L, O, gpu):
        self.Data = Data
        self.E = E
        self.D = D
        self.L = L
        self.O = O
        self.gpu = gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenoisingAutoencoder().to(self.device)
        self.train = self.train()
        

    def train(self):
        # Training loop
        O = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(EPOCH):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            counter = 1
            idx = 10
            for batch in self.Data:
                aug_images, clean_images = batch
                aug_images, clean_images = aug_images.to(self.device), clean_images.to(self.device)  
                
                recon_batch, _, _ = self.model(aug_images)
                loss = self.L.calculate_loss(recon_batch, clean_images)

                O.zero_grad()
                loss.backward()
                O.step()

                running_loss += loss.item()
                
                # s= calculate_ssim(self.model, self.Data, self.device, 'ae')
                if counter == 10:
                    # print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,idx,loss,s))
                    counter = 0
                    idx += 10
            s= calculate_ssim(self.model, self.Data, self.device, 'ae')
            print(f"----- Epoch:{epoch}, Loss:{running_loss}, Similarity:{s}")

            # if epoch%5 == 0:
                # plot_tsne_embeddings(self.model.encoder, self.Data, device, 'AE', epoch)


class VAETrainer:

    def __init__(self, Data, E, D, L, O, gpu):

        self.Data = Data
        self.E = E
        self.D = D
        self.L = L
        self.O = O
        self.gpu = gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VariationalDenoisingAutoencoder().to(self.device)
        self.train = self.train()

    def train(self):
        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        O = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(EPOCH):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            counter = 1
            idx = 10
            for batch in self.Data:
                aug_images, clean_images = batch
                aug_images, clean_images = aug_images.to(self.device), clean_images.to(self.device)  # Move tensors to the same device as the model

                recon_batch, mu, logvar = self.model(aug_images)
                loss = self.L.calculate_loss(recon_batch, clean_images, mu, logvar)
                
                self.O.zero_grad()
                loss.backward()
                self.O.step()

                running_loss += loss.item()

                # s= calculate_ssim(self.model, self.Data, self.device)
                if counter%10 == 0:
                    # print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,idx,loss,s))
                    counter = 0
                    idx += 10
            
            # Print average loss per epoch
            s= calculate_ssim(self.model, self.Data, self.device)
            print(f"----- Epoch:{epoch}, Loss:{running_loss}, Similarity:{s}")

            # if epoch%5 == 0:
                # plot_tsne_embeddings(model.encoder, self.Data, device, 'VAE', epoch)


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    pass

    def from_path(sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    pass

    def from_path(sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass
