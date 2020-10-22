+++
date = "2020-07-24"
title = "Fusion Vision"
showonlyimage = false
draft = false
image = "https://dildehdrg5ol8.cloudfront.net/images/1484-8709d2e4c09315bc96a1a9a1e897d1ac.png"
weight = 2
+++

Providing control over StyleGAN2 to generate new realistic images.
<!--more-->

![img](https://dildehdrg5ol8.cloudfront.net/images/1484-8709d2e4c09315bc96a1a9a1e897d1ac.png)

### What is it?
Fusion Vision is an application that provides creative control over the generative process of StyleGAN2.

### Who is it for?
This application is targeted towards artists who are interested in using the cutting edge AI power. I've also provided jupyter notebooks for those comfortable with code who want to tweak things more.

### The problem
GANs (Generative Adversarial Networks) are well known for their ability to create new images, but they are hard to control. Fusion Vision handles the mathematics behind finding the right controls, so the artist can focus only on what is important to them - translating their ideas into creations.

### The solution
Controls that offer the most degree of freedom in the StyleGAN2 latent space can be found using PCA (Principal Component Analysis). This process is described in depth in the academic paper titled: '[GANSpace: Discovering Interpretable GAN Controls.](https://arxiv.org/abs/2004.02546)' I refined those controls and presented them using a friendly user interface.

#### Github: [https://github.com/sdhnshu/Fusion-Vision](https://github.com/sdhnshu/Fusion-Vision)