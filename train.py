import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from models import Generator
from models import Discriminator
from datasets import ImageDataset

from torch.utils.tensorboard import SummaryWriter

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(G_A2B,G_B2A,D_A,D_B,lr=0.0002,batch_size=32,n_epochs=100):
    # Losses functions
    GAN_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    # Optimizers
    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(),lr=lr, betas=(0.5, 0.999))
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(),lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # Inputs and targets
    input_A = torch.FloatTensor(batch_size, 3, 224, 224).to(device)
    input_B = torch.FloatTensor(batch_size, 3, 224, 224).to(device)
    target_real = Variable(torch.FloatTensor(batch_size).fill_(1.0), requires_grad=False).to(device)
    target_fake = Variable(torch.FloatTensor(batch_size).fill_(0.0), requires_grad=False).to(device)

    # Buffers of previous generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = transforms.Compose([ transforms.Resize(256), 
                    transforms.RandomCrop(224), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
    dataloader = DataLoader(ImageDataset("datasets/summer2winter_yosemite", transforms_=transforms_, unaligned=True), 
                            batch_size=batch_size, shuffle=True, num_workers=4)
    print(len(dataloader))
    writer = SummaryWriter()
    for epoch in range(n_epochs):
        print("Epoch {}/{}:".format(epoch,n_epochs))
        for _,batch in tqdm(enumerate(dataloader)):
            # Get model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            
            ### Training generator G_A2B and G_B2A
            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()
            # Identity loss
            same_B = G_A2B(real_B)
            loss_identity_B = identity_loss(same_B, real_B)*5.0
            same_A = G_B2A(real_A)
            loss_identity_A = identity_loss(same_A, real_A)*5.0
            # GAN loss
            fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_A2B = GAN_loss(pred_fake, target_real)
            fake_A = G_B2A(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_B2A = GAN_loss(pred_fake, target_real)
            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_ABA = cycle_loss(recovered_A, real_A)*10.0
            recovered_B = G_A2B(fake_A)
            loss_cycle_BAB = cycle_loss(recovered_B, real_B)*10.0
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()            
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()

            ### Training Discriminator A 
            optimizer_D_A.zero_grad()
            # Real loss
            pred_real = D_A(real_A)
            loss_D_real = GAN_loss(pred_real, target_real)
            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)     
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = GAN_loss(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            ### Training Discriminator B
            optimizer_D_B.zero_grad()
            # Real loss
            pred_real = D_B(real_B)
            loss_D_real = GAN_loss(pred_real, target_real)
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = GAN_loss(pred_fake, target_fake)
            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()

        # logging
        writer.add_scalar('Loss G', loss_G, epoch)
        writer.add_scalar('loss_G_identity',loss_identity_A + loss_identity_B,epoch)
        writer.add_scalar('loss_G_GAN',loss_GAN_A2B + loss_GAN_B2A,epoch)
        writer.add_scalar('loss_G_cycle',loss_cycle_ABA + loss_cycle_BAB,epoch)
        writer.add_scalar('Loss D', loss_D_A+loss_D_B, epoch)
        writer.add_scalar('Loss D_A', loss_D_A, epoch)
        writer.add_scalar('Loss D_B', loss_D_B, epoch)

        # Save models checkpoints
        torch.save(G_A2B.state_dict(), 'checkpoints/G_A2B_%d.pth'%epoch)
        torch.save(G_B2A.state_dict(), 'checkpoints/G_B2A_%d.pth'%epoch)
        torch.save(D_A.state_dict(), 'checkpoints/D_A_%d.pth'%epoch)
        torch.save(D_B.state_dict(), 'checkpoints/D_B_%d.pth'%epoch)
    writer.close()

#### Helper classses and function
# Class to save the previous buffer
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


if __name__=="__main__":
    # Define Generators and Discriminators
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    
    train(G_A2B,G_B2A,D_A,D_B,batch_size=2,lr=0.0002,n_epochs=200)