# Fire Together Wire Together (FTWT)

Sara Elkerdawy<sup>1</sup>, Mostafa Elhoushi<sup>2</sup>, Hong Zhang<sup>1</sup>, Nilanjan Ray<sup>1</sup>

<sup>1</sup> Computing Science Departement, University of Alberta, Canada\
<sup>2</sup> Toronto Heterogeneous Compilers Lab, Huawei, Canada

Sample training code for CIFAR fo dynamic pruning with self-supervised mask.

[[Project Page](https://selkerdawy.github.io/FTWT/)], [[Paper CVPR22](https://arxiv.org/abs/2110.08232)], [[Poster](docs/assets/img/CVPR22_PosterID178a_PaperID3265.pdf)], [[Video](#)]


<img width="965" alt="image" src="https://user-images.githubusercontent.com/1451293/170155960-9c5e133e-8212-45fd-8a72-6c1fc84ef12d.png">
<figcaption align = "center"><b>FLOPs reduction vs accuracy drop from baselines for various dynamic and static models on ResNet34 ImageNet.</b></figcaption>

## Environment
```
virtualenv .envpy36 -p python3.6 #Initialize environment
source .envpy36/bin/activate
pip install -r req.txt # Install dependencies
```
  
## Train baseline
```
sh job_baseline.sh #You can change model at line 5
```
  
## Train dynamic
```
sh job_dynamic.sh #You can change model at line 5 and threshold at line 40
```

## Results

| `dataset`  | `model`       | `mthresh`   | `mode`      | Accuracy | FLOPS Reduction (%) |
| ---------- | ------------- | ----------- | ----------- | -------- | ------------------- |
| `cifar10`  | `vgg16-bn`    |             |             | 93.82%   | Baseline            |
|            |               | 0.92        | `joint`     | 93.55%   | 65%                 |
|            |               | 0.92        | `decoupled` | 93.73%   | 56%                 |
|            |               | 0.85        | `decoupled` | 93.19%   | 73%                 |
|            |               | 0.88        | `joint`     | 92.65%   | 74%                 |
|            | `resnet56`    |             |             | 93.66%   | Baseline            |
|            |               | 0.80        | `decoupled` | 92.63%   | 66%                 |
|            |               | 0.88        | `joint`     | 92.28%   | 54%                 |
|            | `mobilenetv1` |             |             | 90.89%   | Baseline            |
|            |               | 1.00        | `decoupled` | 91.06%   | 78%                 |
|            |               | 1.00        | `joint`     | 91.21%   | 78%                 |
| `imagenet` | `resnet34`    |             |             | 73.30%   | Baseline            |
|            |               | 0.97        | `decoupled` | 73.25%   | 25.86%              |
|            |               | 0.95        | `decoupled` | 72.79%   | 37.77%              |
|            |               | 0.93        | `decoupled` | 72.17%   | 47.42%              |
|            |               | 0.92        | `decoupled` | 71.71%   | 52.24%              |
|            | `resnet18`    |             |             | 69.76%   | Baseline            |
|            |               | 0.91        | `decoupled` | 67.49%   | 51.56%              |
|            | `mobilenetv1` |             |             | 69.57%   | Baseline            |
|            |               | 1.00        | `decoupled` | 69.66%   | 41.07%              |


## Citation
```
@InProceedings{elkerdawy2022fire,
    author    = {Elkerdawy, Sara and Elhoushi, Mostafa and Zhang, Hong and Ray, Nilanjan},
    title     = {Fire Together Wire Together: A Dynamic Pruning Approach with Self-Supervised Mask Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
}
```
