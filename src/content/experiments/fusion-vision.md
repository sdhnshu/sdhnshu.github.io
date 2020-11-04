+++
date = "2020-08-18"
title = "Fusion Vision"
showonlyimage = false
draft = false
image = "https://dildehdrg5ol8.cloudfront.net/images/1484-8709d2e4c09315bc96a1a9a1e897d1ac.png"
weight = 1
+++

Providing control over StyleGAN2 to generate new realistic images.
<!--more-->

![img](https://dildehdrg5ol8.cloudfront.net/images/1484-8709d2e4c09315bc96a1a9a1e897d1ac.png)

- Code on Github: [github.com/sdhnshu/Fusion-Vision](https://github.com/sdhnshu/Fusion-Vision)

### Intro
Fusion Vision is an Data Manipulation Tool that provides you creative control over the generative process of StyleGAN2.

### Who should use this tool?

- Artists who want to use StyleGAN's potential to generate art.
- Sketch Artists in police stations.
- Those wanting to preserve the privacy of people in their datasets.
- Anyone who wants to try something new!

### The problem
GANs (Generative Adversarial Networks) are well known for their ability to create new images, but they are hard to control. Fusion Vision handles the mathematics behind finding the right controls, so the artist can focus on what is important to them - translating their ideas into creations.

### The solution
Controls that offer the most degree of freedom in the StyleGAN2 latent space can be found using PCA (Principal Component Analysis). This process is described in depth in the academic paper titled: '[GANSpace: Discovering Interpretable GAN Controls.](https://arxiv.org/abs/2004.02546)'

I've refined those controls and presented them using a friendly user interface. I've also provided jupyter notebooks for those comfortable with code who want to tweak things more.

__Caution:__ The project is still in Beta phase. If you find any bugs or have issues, please open up an issue [here](https://github.com/sdhnshu/Fusion-Vision/issues). Any pull requests are welcome.