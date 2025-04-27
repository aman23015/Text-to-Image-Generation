import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Critic, Generator, initialize_weights

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITER = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/",train=True,transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(),lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(),lr=LEARNING_RATE)

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)

        for _ in range(CRITIC_ITER):
            noise = torch.randn(BATCH_SIZE,Z_DIM,1,1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1) # so that we only get one value rather than NX1x1x1.
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ## Train Generator: min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ## Print losses
        if batch_idx %100 ==0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \ "
                f"Loss D:{loss_critic:.4f}, Loss G:{loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                data = real[:32]
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Image", img_grid_fake,global_step=step
                )

                writer_real.add_image(
                    "Mnist real Image", img_grid_real,global_step=step
                )
            step += 1
            

