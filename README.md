# Cat Image Generation Using GANs

## Overview
This project demonstrates the generation of cat images using Generative Adversarial Networks (GANs). The model includes both the Generator and Discriminator networks, trained to create realistic images from random noise. The training process leverages two types of loss functions: the original GAN loss and the Least Squares GAN loss.

## Loss Functions Implemented

### GAN Loss
The standard GAN loss functions used in this project are:

- **Generator Loss**:
  $$
  \ell_G  =  -\mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]
  $$

- **Discriminator Loss**:
  $$
  \ell_D = -\mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] - \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]
  $$

### Least Squares GAN Loss
An alternative loss function is implemented as described in the [Least Squares GAN paper](https://arxiv.org/abs/1611.04076):

- **Generator Loss**:
  $$
  \ell_G  =  \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[\left(D(G(z))-1\right)^2\right]
  $$

- **Discriminator Loss**:
  $$
  \ell_D = \frac{1}{2}\mathbb{E}_{x \sim p_\text{data}}\left[\left(D(x)-1\right)^2\right] + \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[ \left(D(G(z))\right)^2\right]
  $$

## How GANs Work
Generative Adversarial Networks (GANs) consist of two neural networks: the Generator and the Discriminator. The Generator creates fake images from random noise, while the Discriminator tries to distinguish between real and fake images. The two networks are trained simultaneously: the Generator aims to improve its ability to produce realistic images, while the Discriminator improves its ability to identify fake images.

## Images
Below are some examples of generated cat images at various stages of training:

![Cat Images](link-to-your-images-folder)

Feel free to explore the code and modify it as per your requirements.

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the training script to start training the GAN.

For more details, refer to the code and comments within the repository.
