# Fire Together Wire Together
Sample training code for CIFAR fo dynamic pruning with self-supervised mask
[[Project Page](https://selkerdawy.github.io/FTWT/)], [[Paper](https://arxiv.org/abs/2110.08232)], [[Poster](assets/img/CVPR22_PosterID178a_PaperID3265.pdf)], [[Video](#)]


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
