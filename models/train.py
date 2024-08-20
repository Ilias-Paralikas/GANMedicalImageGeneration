import torch
import torch.nn as nn
from utils import generate_fixed_random_tensor

def train_GAN(generator,
              discriminator,
              gen_optimizer,
              disc_optimizer,
              loss_function,
              dataloader,
              epochs,
              generator_epochs,
              device,
              save_frequency=1,
              versbose_frequency=1000,
              static=False):
  
    def generate_noise(batch_size,noise_shape,static):
        if static :
            noise = generate_fixed_random_tensor(0,noise_shape,device=device)
        else:
            noise  = torch.randn(batch_size, *noise_shape, device=device)
            
        return noise
    
    noise_shape = generator.noise_shape


    for epoch in range(epochs):
        epoch_generator_loss =0 
        epoch_discriminator_loss = 0
        
        for i,real_images in enumerate(dataloader):

            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            noise = generate_noise(batch_size,noise_shape,static)
            fake_images = generator(noise)

            discriminator_real_output  = discriminator(real_images).view(-1)
            discriminator_real_loss = loss_function(discriminator_real_output,torch.ones_like(discriminator_real_output))
            
            discriminator_fake_output = discriminator(fake_images).view(-1)
            discriminator_fake_loss = loss_function(discriminator_fake_output,torch.zeros_like(discriminator_fake_output))

            discriminator_loss = (discriminator_real_loss+discriminator_fake_loss)/2
            disc_optimizer.zero_grad()
            discriminator_loss.backward()
            disc_optimizer.step()
            
            # GENERATOR TRAIN
            for _ in range(generator_epochs):
                noise = generate_noise(batch_size,noise_shape,static)
                fake_images = generator(noise)
                fake_output=discriminator(fake_images).view(-1)
                generator_loss =  loss_function(fake_output,torch.ones_like(fake_output))
                gen_optimizer.zero_grad()
                generator_loss.backward()
                gen_optimizer.step()
            
            epoch_generator_loss  +=generator_loss.item()
            epoch_discriminator_loss +=discriminator_loss.item()
            
            if i % save_frequency ==0 :
                torch.save(generator.state_dict(), generator.path)
                torch.save(discriminator.state_dict(), discriminator.path)
            if i % versbose_frequency ==0 :
                if i !=0:
                    print(f"Sample: {i} Generator Loss: {generator_loss.item()} Discriminator Loss: {discriminator_loss.item()}")

        print(f"Epoch :{epoch+1}/{epochs}\tGenerator Loss: {epoch_generator_loss/len(dataloader)} \tDiscriminator Loss: {epoch_discriminator_loss/len(dataloader)}")      

