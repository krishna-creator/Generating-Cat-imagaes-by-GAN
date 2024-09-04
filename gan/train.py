import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=1000, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
            # Zero gradients for both networks
            D_solver.zero_grad()
            G_solver.zero_grad()
            
            # Train discriminator with real images
            logits_real = D(real_images)
            noise = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(noise)
            logits_fake = D(fake_images.detach())
            
            d_error = discriminator_loss(logits_real, logits_fake)
            d_error.backward()
            D_solver.step()
            
            # Train generator
            G_solver.zero_grad()
            noise = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(noise)
            logits_fake = D(fake_images)
            
            g_error = generator_loss(logits_fake)
            g_error.backward()
            G_solver.step()
            
            # Logging and visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_error.item(), g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.show()
                print()
            iter_count += 1
