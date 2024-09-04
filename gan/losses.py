import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss_real = bce_loss(logits_real, torch.ones_like(logits_real))
    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    loss = loss_real + loss_fake
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake))
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the LSGAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss_real = torch.mean((scores_real - 1) ** 2)
    loss_fake = torch.mean(scores_fake ** 2)
    loss = 0.5 * (loss_real + loss_fake)

    return loss
    

def ls_generator_loss(scores_fake):
    """
    Computes the LSGAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)

    return loss
