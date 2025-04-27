import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from utilities import*


##+++++++++ Dataset and Dataloader ++++++++++++##

train_dataset = FlowersDataset("train_embeddings.pt")
test_dataset = FlowersDataset("test_embeddings.pt")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16, shuffle=False)

##+++++++++++ Architectures +++++++++++++++++++##

class SourceEncoder(nn.Module):
    def __init__(self, encoded_dim=128):
        super(SourceEncoder, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.fc = nn.Linear(512, encoded_dim)  # Reduce output to 128-dim

    def forward(self, x):
        return self.encoder(x)


class TargetGenerator(nn.Module):
    def __init__(self, encoded_dim=128, text_dim=64, noise_dim=100, img_channels=3):
        super(TargetGenerator, self).__init__()
        input_dim = encoded_dim + text_dim  # Conditioned on image & text representation

        self.model = nn.Sequential(
            nn.Linear(input_dim + noise_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, img_channels * 32 * 32),  # Output 32x32 image
            nn.Tanh()
        )

    def forward(self, img_repr, text_repr, noise):
        text_repr = torch.mean(text_repr,dim=1) #[batch-size,embedding_dim]
        x = torch.cat((img_repr, text_repr, noise), dim=1)
        x = self.model(x)
        x = x.view(-1, 3, 32, 32)  # Reshape to image dimensions
        return x


class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),  # Output a single score
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

##++++++++++++++++++++++++++++++++++++++++++++##

adversarial_loss = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder = SourceEncoder().to(device)
generator = TargetGenerator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

##+++++++++++++++++++ Training loop ++++++++++++##

num_epochs = 100
for epoch in range(num_epochs):
    for real_imgs, text_embeddings in train_loader:
        real_imgs = real_imgs.to(device)
        text_embeddings = text_embeddings.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(real_imgs)
        real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds))
        
        noise = torch.randn(real_imgs.size(0), 100).to(device)
        img_repr = encoder(real_imgs)
        fake_imgs = generator(img_repr, text_embeddings, noise)

        fake_preds = discriminator(fake_imgs.detach())
        fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_preds = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}: D Loss {d_loss.item()}, G Loss {g_loss.item()}")
    if (epoch + 1) % 10 == 0:
        torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
        print(f"Saved models at epoch {epoch+1}!")

def generate_images(generator, encoder, text_embeddings, num_images=5):
    noise = torch.randn(num_images, 100).to(device)
    with torch.no_grad():
        img_repr = torch.randn(num_images, 128).to(device)  # Use random embeddings for unseen classes
        generated_imgs = generator(img_repr, text_embeddings, noise)
    return generated_imgs

test_text_embeddings = torch.randn(5, 64).to(device)
gen_images = generate_images(generator, encoder, test_text_embeddings)
