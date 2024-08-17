import torch
import torch.nn as nn

def train_GAN(generator,
              discriminator,
              dataloader,
              gen_optim,
              disc_optim,
              loss_fn,
              epochs=1,
              device= 'cpu',
              save_frequency=10,
              verbose_frequency=10,
              static_input= False):
    def test_compatibility(generator, discriminator,dataloader,noise_shape,device):
      single_element= next(iter(dataloader))[0]
      noise = torch.randn(1,*noise_shape).to(device)
      fake_image = generator(noise)
      output = discriminator(fake_image)        
      assert single_element.shape == fake_image.squeeze(0).shape
      assert output.shape == torch.Size([1, 1])
        
    noise_shape = generator.noise_shape

    test_compatibility(generator, discriminator,dataloader,noise_shape,device)
        
    generator_loss_history = []
    discriminator_loss_history = []


    real_label = 1
    fake_label = 0
    print('Started Training')
    if static_input:
      torch.manual_seed(0)
      # If using CUDA
      if torch.cuda.is_available():
          torch.cuda.manual_seed(0)
          torch.cuda.manual_seed_all(0) 
      static_noise  = torch.randn(1, *noise_shape, device=device)

    for epoch in range(epochs):
        discriminator_epoch_loss = 0
        generator_epoch_loss = 0
        for i,real_images in enumerate(dataloader):
          disc_optim.zero_grad()
          
          real_images = real_images.to(device)
          batch_size = real_images.shape[0]

          label = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
          
          if static_input:
            noise  = static_noise
          else:
            noise  = torch.randn(batch_size, *noise_shape, device=device)

          fake_images = generator(noise)
          output = discriminator(fake_images.detach()).view(-1)

          fake_loss  = loss_fn(output,label)
          fake_loss.backward()
      

          label.fill_(real_label)
          output = discriminator(real_images).view(-1)
          # # calculate the loss between the real and the generated labes
          real_loss = loss_fn(output, label)
          real_loss.backward()

          disc_optim.step()

          disc_loss = real_loss  + fake_loss
          # print('Discriminaor Loss :',disc_loss.item())
          discriminator_epoch_loss += disc_loss.item()

          gen_optim.zero_grad()
          label.fill_(real_label)  
          output = discriminator(fake_images).view(-1)
          gen_loss = loss_fn(output,label)
          gen_loss.backward()
          gen_optim.step()
          
          # print('Generator Loss :',gen_loss.item())
          generator_epoch_loss += gen_loss.item()
          
          if i % save_frequency ==save_frequency-1:
            torch.save(generator.state_dict(), generator.path)
            torch.save(discriminator.state_dict(), discriminator.path)
            print('Model Saved')
          if i % verbose_frequency == verbose_frequency-1:
            if i !=0:
              print(f"Generator Loss: {generator_epoch_loss/ i} Discriminator Loss: {discriminator_epoch_loss/ i}")
        discriminator_loss_history.append(discriminator_epoch_loss)
        generator_loss_history.append(generator_epoch_loss)
        print(f"Epoch {epoch} Generator Loss: {generator_epoch_loss/ len(dataloader)} Discriminator Loss: {discriminator_epoch_loss/ len(dataloader)}")
      
    torch.save(generator.state_dict(), generator.path)
    torch.save(discriminator.state_dict(), discriminator.path)
    print('Model Saved')
    return generator_loss_history, discriminator_loss_history
  
  

