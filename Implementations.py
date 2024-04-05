from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models.mobilenetv2 as mobilenetv2
import torchvision.datasets.folder as folder
from torchvision.datasets.folder import default_loader
import torchvision.utils as vutils
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

EPOCH = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCH = 50
SAVEPATH = os.getcwd()

aug_path = '../Data/aug'
clean_path = '../Data/clean'

clean_images_dict = {}
for clean_image_name in os.listdir(clean_path):
    label = int(clean_image_name.split('_')[-1].split('.')[0])
    if label not in clean_images_dict:
        clean_images_dict[label] = []
    clean_images_dict[label].append(os.path.join(clean_path, clean_image_name))

aug_images = os.listdir(aug_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using MobileNetV2
model = mobilenetv2.mobilenet_v2(pretrained=True).to(device)  # Move model to device
model.classifier = nn.Identity()  # Remove the last fully connected layer
input_size = 224  # MobileNetV2 default input size

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # Resize to fit MobileNetV2 input size
    transforms.ToTensor(),
])

def extract_embeddings(path):
    embeddings = {}
    lst = os.listdir(path)
    for idx in range(len(lst)):
        if idx % 1000 == 0:
            print(idx, '/', len(lst))
        img_name = lst[idx]
        img_path = os.path.join(path, img_name)
        img = default_loader(img_path)
        img_tensor = transform(img).unsqueeze(0).to(device)  # Move tensor to device
        with torch.no_grad():
            embedding = model(img_tensor)
        embeddings[img_name] = embedding.squeeze(0)
    return embeddings

def extract_aug_embeddings():
    embeddings = {}
    for idx in range(len(aug_images)):
        if idx%1000 == 0:
            print(idx,'/', len(aug_images))
        aug_filename = aug_images[idx]
        aug_img_path = os.path.join(aug_path, aug_filename)
        aug_img = default_loader(aug_img_path)
        if transform:
            aug_img = transform(aug_img).to(device)  # Move tensor to device
        aug_embedding = extract_embedding(aug_img)
        embeddings[aug_filename] = aug_embedding
    return embeddings

def extract_embedding(img):
    img_tensor = img.unsqueeze(0).to(device)  # Move tensor to device
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze(0)

def generate_mappings(aug_embeddings, clean_embeddings):
    mappings = {}
    for idx, aug_filename in enumerate(aug_images):
        if idx%1000 == 0:
            print(idx, '/', len(aug_images))
        aug_label = int(aug_filename.split('_')[-1].split('.')[0])
        aug_embedding = aug_embeddings[aug_filename]
        aug_embedding_batch = aug_embedding.unsqueeze(0)
        aug_embedding_batch = aug_embedding_batch.expand(len(clean_images_dict[aug_label]), -1)
        clean_embeddings_list = [clean_embeddings[os.path.basename(clean_img_path)] for clean_img_path in clean_images_dict[aug_label]]
        clean_embeddings_tensor = torch.stack(clean_embeddings_list)
        similarities = F.cosine_similarity(aug_embedding_batch.to(device), clean_embeddings_tensor.to(device), dim=1)
        max_similarity_idx = torch.argmax(similarities)
        max_clean_img_path = list(clean_images_dict[aug_label])[max_similarity_idx]

        mappings[idx] = max_clean_img_path

    return mappings

def perform():
    clean_embeddings = extract_embeddings(clean_path)
    aug_embeddings = extract_aug_embeddings()
    mappings = generate_mappings(aug_embeddings, clean_embeddings)
    return mappings

class AlteredMNIST(Dataset):
    def __init__(self, transform=None):
        self.aug_path = aug_path
        self.clean_path = clean_path
        self.transform = transform
        self.aug_images = os.listdir(aug_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mappings = perform()

    def __len__(self):
        return len(self.aug_images)

    def __getitem__(self, idx):
        max_clean_img_path = self.mappings[idx]
        aug_filename = self.aug_images[idx]
        aug_img_path = os.path.join(self.aug_path, aug_filename) 
        aug_label = int(aug_filename.split('_')[-1].split('.')[0])

        aug_img = folder.default_loader(aug_img_path)
        clean_img = folder.default_loader(max_clean_img_path)

        aug_img = self.transforms(aug_img)
        clean_img = self.transforms(clean_img)

        return aug_img, clean_img, aug_label
    

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
            ResNetBlock(64, 64)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_mu = nn.Linear(64*14*14, 64)
        self.fc_logvar = nn.Linear(64*14*14, 64)
        self.fc_labels = nn.Linear(10, 64)
        self.fc_class_mu = nn.Linear(64, 64)
        self.fc_class_logvar = nn.Linear(64, 64)

    def forward(self, x, labels=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if labels is None:
            mu = self.fc_mu(out)
            logvar = self.fc_logvar(out)
            return mu, logvar
        else:
            mu = self.fc_mu(out)
            logvar = self.fc_logvar(out)
            labels_one_hot = F.one_hot(labels, num_classes=10).float()
            labels_embedding = labels_one_hot @ self.fc_labels.weight.T
            mu += labels_embedding
            logvar += labels_embedding
            class_mu = self.fc_class_mu(labels_embedding)
            class_logvar = self.fc_class_logvar(labels_embedding)
            return mu, logvar, class_mu, class_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(64, 64 * 14 * 14)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.conv_transpose2 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.num_classes = 10

    def forward(self, x, labels=None):
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
        for aug_images, _, _ in dataloader:
            aug_images = aug_images.to(device)
            mu, _ = encoder(aug_images)
            all_embeddings.append(mu.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings)

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=5, n_iter=250, early_exaggeration=12)
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

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class AETrainer:
    def __init__(self, Data, E, D, L, O, gpu):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Data = Data
        self.E = E.to(device)
        self.D = D.to(device)
        self.L = L
        self.O = O
        self.gpu = gpu
        self.train = self.train()
        

    def train(self):
        # Training loop
        for epoch in range(EPOCH):
            running_loss = 0.0

            counter = 1
            idx = 10
            recon_images_list = []
            clean_images_list = []
            ssim_scores = []  # List to store SSIM scores for each batch
            for batch in self.Data:
                aug_images, clean_images, _ = batch
                aug_images, clean_images = aug_images.to(self.device), clean_images.to(self.device)  
                
                encoded = self.E(aug_images)
                recon_images = self.D(encoded)
                recon_images_list.append(recon_images)
                clean_images_list.append(clean_images)
                loss = self.L.calculate_loss(recon_images, clean_images)

                self.O.zero_grad()
                loss.backward()
                self.O.step()

                running_loss += loss.item()
                
                if counter == 10:
                    recon_images_batch = torch.cat(recon_images_list, dim=0)
                    clean_images_batch = torch.cat(clean_images_list, dim=0)
                    recon_images_batch = recon_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    clean_images_batch = clean_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    batch_ssim = ssim(recon_images_batch, clean_images_batch, multichannel=True)
                    ssim_scores.append(batch_ssim)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,idx,loss,batch_ssim))
                    counter = 0
                    idx += 10
                    recon_images_list = []
                    clean_images_list = []

                counter += 1

            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            print(f"----- Epoch:{epoch}, Loss:{running_loss}, Similarity:{avg_ssim}")

            if (epoch+1)%10 == 0:
                plot_tsne_embeddings(self.E, self.Data, device, 'AE', epoch)
            
        torch.save(self.E.state_dict(), 'aeE.pth')
        torch.save(self.D.state_dict(), 'aeD.pth')

class VAETrainer:

    def __init__(self, Data, E, D, L, O, gpu):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Data = Data
        self.E = Encoder().to(self.device)
        self.D = Decoder().to(self.device)
        self.L = L
        self.O = torch.optim.Adam(ParameterSelector(self.E, self.D), lr=LEARNING_RATE)
        self.gpu = gpu
        self.train = self.train()

    def train(self):
        for epoch in range(EPOCH):
            running_loss = 0.0
            counter = 1
            idx = 10
            recon_images_list = []
            clean_images_list = []
            ssim_scores = []  # List to store SSIM scores for each batch
            for batch in self.Data:
                aug_images, clean_images, _ = batch
                aug_images, clean_images = aug_images.to(self.device), clean_images.to(self.device)  # Move tensors to the same device as the model

                mu, logvar = self.E(aug_images)
                z = reparameterize(mu, logvar)
                recon_batch = self.D(z)
                loss = self.L.calculate_loss(recon_batch, clean_images, mu, logvar)

                loss = torch.clamp(loss, max=1000000000000000)

                recon_images_list.append(recon_batch)
                clean_images_list.append(clean_images)
                
                self.O.zero_grad()
                loss.backward()
                self.O.step()

                running_loss += loss.item()

                if counter == 10:
                    recon_images_batch = torch.cat(recon_images_list, dim=0)
                    clean_images_batch = torch.cat(clean_images_list, dim=0)
                    recon_images_batch = recon_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    clean_images_batch = clean_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    batch_ssim = ssim(recon_images_batch, clean_images_batch, multichannel=True)
                    ssim_scores.append(batch_ssim)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,idx,loss,batch_ssim))
                    counter = 0
                    idx += 10
                    recon_images_list = []
                    clean_images_list = []

                counter += 1

            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            print(f"----- Epoch:{epoch}, Loss:{running_loss}, Similarity:{avg_ssim}")

            if (epoch+1)%10 == 0:
                plot_tsne_embeddings(self.E, self.Data, device, 'VAE', epoch)
            
        torch.save(self.E.state_dict(), 'vaeE.pth')
        torch.save(self.D.state_dict(), 'vaeD.pth')
            

class CVAELossFn:
    def calculate_loss(self, recon_batch, clean_images, mu, logvar, labels, class_means, class_logvars):
        MSE = F.mse_loss(recon_batch, clean_images, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        class_KLD = -0.5 * torch.sum(1 + class_logvars - class_means.pow(2) - class_logvars.exp())
        return MSE + KLD + class_KLD

class CVAE_Trainer:
    def __init__(self, Data, E, D, L, O):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Data = Data
        self.E = Encoder().to(self.device)
        self.D = Decoder().to(self.device)
        self.L = L
        self.O = torch.optim.Adam(ParameterSelector(self.E, self.D), lr=LEARNING_RATE)
        self.train()

    def train(self):
        for epoch in range(EPOCH):
            running_loss = 0.0
            counter = 1
            idx = 10
            recon_images_list = []
            clean_images_list = []
            ssim_scores = []  # List to store SSIM scores for each batch
            for i, batch in enumerate(self.Data):
                aug_images, clean_images, labels = batch
                aug_images, clean_images, labels = aug_images.to(self.device), clean_images.to(self.device), labels.to(self.device)

                mu, logvar, class_means, class_logvars = self.E(aug_images, labels)
                z = reparameterize(mu, logvar)
                recon_batch = self.D(z, labels)
                recon_images_list.append(recon_batch)
                clean_images_list.append(clean_images)
                loss = self.L.calculate_loss(recon_batch, clean_images, mu, logvar, labels, class_means, class_logvars)
                loss = torch.clamp(loss, max=1000000000000)

                self.O.zero_grad()
                loss.backward()
                self.O.step()

                running_loss += loss.item()

                if counter == 10:
                    recon_images_batch = torch.cat(recon_images_list, dim=0)
                    clean_images_batch = torch.cat(clean_images_list, dim=0)
                    recon_images_batch = recon_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    clean_images_batch = clean_images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy and reshape
                    batch_ssim = ssim(recon_images_batch, clean_images_batch, multichannel=True)
                    ssim_scores.append(batch_ssim)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,idx,loss,batch_ssim))
                    counter = 0
                    idx += 10
                    recon_images_list = []
                    clean_images_list = []

                counter += 1

            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            print(f"----- Epoch:{epoch}, Loss:{running_loss}, Similarity:{avg_ssim}")

            if (epoch+1)%10 == 0:
                plot_tsne_embeddings(self.E, self.Data, device, 'VAE', epoch)

        torch.save(self.E.state_dict(), 'cvaeE.pth')
        torch.save(self.D.state_dict(), 'cvaeD.pth')

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()

def caclulate_one(aug_img, clean_img, encoder, decoder, device, name, what):
    aug_img = aug_img.unsqueeze(0).to(device)
    clean_img = clean_img.unsqueeze(0).to(device)
    aug_img = aug_img.to(device)
    clean_img = clean_img.to(device)
    if name == 'ae':
        encoded = encoder(aug_img)
        recon_img = decoder(encoded)
    elif name == 'cvae':
        mu, logvar, _, _ = encoder(aug_img)
        z = reparameterize(mu, logvar)
        recon_img = decoder(z)
    else:
        mu, logvar = encoder(aug_img)
        z = reparameterize(mu, logvar)
        recon_img = decoder(z)

    # Convert recon_img to grayscale
    recon_img_gray = 0.299 * recon_img[:, 0, :, :] + 0.587 * recon_img[:, 1, :, :] + 0.114 * recon_img[:, 2, :, :]
    recon_img_gray = recon_img_gray.unsqueeze(1)
    
    # Convert clean_img to grayscale
    clean_img_gray = 0.299 * clean_img[:, 0, :, :] + 0.587 * clean_img[:, 1, :, :] + 0.114 * clean_img[:, 2, :, :]
    clean_img_gray = clean_img_gray.unsqueeze(1)
    
    # Reshape to 1x28x28
    recon_img_gray = recon_img_gray.view(1, 28, 28)
    clean_img_gray = clean_img_gray.view(1, 28, 28)

    recon_img_gray = recon_img_gray.to('cpu')
    clean_img_gray = clean_img_gray.to('cpu')

    if what=='SSIM':
        return structure_similarity_index(recon_img_gray, clean_img_gray)
    elif what=='PSNR':
        return peak_signal_to_noise_ratio(recon_img_gray, clean_img_gray)

class AE_TRAINED:

    def __init__(self, gpu) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.E = Encoder().to(self.device)
        self.D = Decoder().to(self.device)
        self.E.load_state_dict(torch.load('aeE.pth')) 
        self.D.load_state_dict(torch.load('aeD.pth')) 

    def from_path(self, sample, original, type):
        if type == 'SSIM':
            return caclulate_one(sample, original, self.E, self.D, self.device, 'ae', 'SSIM')
        else:
            return caclulate_one(sample, original, self.E, self.D, self.device, 'ae', 'PSNR')

class VAE_TRAINED:
    def __init__(self, gpu) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.E = Encoder().to(self.device)
        self.D = Decoder().to(self.device)
        self.E.load_state_dict(torch.load('vaeE.pth')) 
        self.D.load_state_dict(torch.load('vaeD.pth')) 

    def from_path(self, sample, original, type):
        if type == 'SSIM':
            return caclulate_one(sample, original, self.E, self.D, self.device, 'vae', 'SSIM')
        else:
            return caclulate_one(sample, original, self.E, self.D, self.device, 'vae', 'PSNR')

class CVAE_Generator:
    def __init__(self, gpu=False):
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.E = Encoder().to(self.device)
        self.D = Decoder().to(self.device)
        self.E.load_state_dict(torch.load('cvaeE.pth', map_location=self.device))
        self.D.load_state_dict(torch.load('cvaeD.pth', map_location=self.device))

    def generate_image(self, digit, save_path):
        # Generate a one-hot vector representing the digit class
        label_tensor = torch.tensor([digit]).to(self.device)
        label_onehot = F.one_hot(label_tensor, num_classes=10).float()

        # Generate a random latent vector
        latent_vector = torch.randn(1, 64).to(self.device)

        # Pass through decoder
        with torch.no_grad():
            generated_image = self.D(latent_vector, label_onehot)

        # Save the generated image
        vutils.save_image(generated_image, save_path)

    def save_image(self, digit, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(24):
            self.generate_image(digit, os.path.join(save_path, f'{digit}_{i}.png'))