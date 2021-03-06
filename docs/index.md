## Fire Together Wire Together: A Dynamic Pruning Approach with Self-Supervised Mask Prediction

[[Paper](https://arxiv.org/abs/2110.08232)], [[Poster](assets/img/CVPR22_PosterID178a_PaperID3265.pdf)], [[Video](#)]


### Abstract

Dynamic model pruning is a recent direction that allows for the inference of a different sub-network for each input sample during deployment. However, current dynamic methods rely on learning a continuous channel gating through regularization by inducing sparsity loss. This formulation introduces complexity in balancing different losses (e.g task loss, regularization loss). In addition, regularization based methods lack transparent tradeoff hyperparameter selection to realize a computational budget. Our contribution is two-fold: 1) decoupled task and pruning losses. 2) Simple hyperparameter selection that enables FLOPs reduction estimation before training. Inspired by the Hebbian theory in Neuroscience: “neurons that fire together wire together”, we propose to predict a mask to process k filters in a layer based on the activation of its previous layer. We pose the problem as a self-supervised binary classification problem. Each mask predictor module is trained to predict if the log-likelihood for each filter in the current layer belongs to the top-k activated filters. The value k is dynamically estimated for each input based on a novel criterion using the mass of heatmaps. We show experiments on several neural architectures, such as VGG, ResNet and MobileNet on CIFAR and ImageNet datasets. On CIFAR, we reach similar accuracy to SOTA methods with 15% and 24% higher FLOPs reduction. Similarly in ImageNet, we achieve lower drop in accuracy with up to 13% improvement in FLOPs reduction.

### Motivation

Lightweight model perform fairly good in most cases and there tends to be a diminishing returns to adding more FLOPs. Noted in the figure, we gain around 2% accuracy increase with double the computation. 

<p align="center">
  <img src="assets/img/motivation.png" alt="Motivation" width="700"/>
</p>

**Question**: Can we only enable the neurons required for each image sample?
Biologically inspired, for different images we look at, different neurons in our brain are fired.

We propose a dynamic inference method to compute different sub-network based on the input samples. Each layer is equipped by a decision gate to select few filters to apply per sample.

### Proposed Method

<p align="center">
  <img src="assets/img/method.png" alt="Pipeline" width="900"/>
</p>

- We propose a novel decision gating loss formulation with self-supervised ground truth mask generation that is stochastic gradient descent (SGD) friendly and decoupled from task loss. Unlike other dynamic inference training methods, a regularization loss is jointly trained with task loss to learn the decision gating. Regularization loss can be hard to tune as pruning ratio increases due to multi-loss gradient interference.
- **During training**, we rank layer’s output features and push the decision gating to predict the **top-k** highly activated features. Top-k is selected based on a hyperparameter **r**.
- **During inference**, we use the binary prediction output from the learned decision gate to perform
handful of filters from the layer based on the input.


### Results

<p align="center">
  <img src="assets/img/results1.png" alt="Results1" width="700"/>
</p>

Our method (FTWT) achieves higher accuracy on similar accuracy drop, especially on high pruning ratio.

<p align="center">
  <img src="assets/img/res2.png" alt="Results2" width="700"/>
</p>

**Did we satisfy the motivation?** 




