import torch
import torch.nn as nn

import torch
import torch.nn as nn

def train_GAN(generator,
              discriminator,
              dataloader,
              noise_shape,
              epochs=1,
              gen_optim =None,
              gen_lr = 0.0002,
              disc_optim=None,
              disc_lr = 0.0002,
              loss_fn=None,
              device= 'cpu'):
    def test_compatibility(generator, discriminator,dataloader,noise_shape,device):
      single_element= next(iter(dataloader))[0]
      noise = torch.randn(1,*noise_shape).to(device)
      fake_image = generator(noise)
      output = discriminator(fake_image)        
      assert single_element.shape == fake_image.squeeze(0).shape
      assert output.shape == torch.Size([1, 1])
        
    test_compatibility(generator, discriminator,dataloader,noise_shape,device)
    gen_optim =gen_optim(generator.parameters(), lr=gen_lr)
    disc_optim = disc_optim(discriminator.parameters(), lr=disc_lr)
    loss_fn = loss_fn()
        
    gen_losses = []
    disc_losses = []


    real_label = 1
    fake_label = 0
    for epoch in range(epochs):
        real_images_loss = 0
        fake_images_loss = 0
        generator_loss = 0
        for real_images in dataloader:
            # train discriminator on real image
            # place the real images on the device
            real_images = real_images.to(device)
            # zero the grad
            disc_optim.zero_grad()
            # get the batch size
            batch_size = real_images.shape[0]
            # create a label array with 1s, as we are talking about real images
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # get the dicsriminator output
            output = discriminator(real_images).view(-1)
            # calculate the loss between the real and the generated labes
            disc_loss = loss_fn(output, label)
            disc_loss.backward()
            real_images_loss += disc_loss.mean().item()
            # train the discriminator on fake images
            noise  = torch.randn(batch_size, *noise_shape, device=device)
            
            # get the fake images
            fake_images = generator(noise)
            # fill the label array with 0s
            label.fill_(fake_label)
            # get the discriminator output
            output = discriminator(fake_images.detach()).view(-1)
            # calculate the loss
            disc_loss = loss_fn(output, label)
            disc_loss.backward()
            fake_images_loss += disc_loss.mean().item()
            disc_optim.step()
        
            # train the generaotr on the output of the discriminator
            gen_optim.zero_grad()
            # the idea is that we want the output of the discriminator to be 1, 
            # so it is fooled and thinks that these are real images, so we 
            label.fill_(real_label)
            output = discriminator(fake_images).view(-1)
            gen_loss = loss_fn(output, label)
            gen_loss.backward()
            generator_loss += gen_loss.mean().item()
            gen_optim.step()
            
        gen_losses.append(generator_loss)
        disc_losses.append(real_images_loss + fake_images_loss)
        print(f"Epoch {epoch} Generator Loss: {generator_loss} Discriminator Loss: {real_images_loss + fake_images_loss}")
        
        torch.save(generator.state_dict(), generator.path)
        torch.save(discriminator.state_dict(), discriminator.path)

    return gen_losses, disc_losses