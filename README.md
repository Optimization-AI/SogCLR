# SogCLR PyTorch Implementation

In this repo, we show how to train a self-supervised model by using [Global Contrastive Loss](https://arxiv.org/abs/2202.12387) (GCL) on [ImageNet](https://image-net.org/). The original GCL was implementated in Tensorflow and run in TPUs [here](https://github.com/Optimization-AI/SogCLR/tree/Tensorflow). This repo **re-implements** GCL in PyTorch based on [moco's](https://github.com/facebookresearch/moco) codebase. We recommend users to run this codebase on GPU-enabled environments, such as [Google Cloud](https://cloud.google.com/), [AWS](https://aws.amazon.com/).

## What's new
- 2023.03.05 Fixed `RuntimeError` related to variable `u`
- 2023.03.05 Fixed `AttributeError` related to `margin`

## Installation

#### git clone
```bash
git clone https://github.com/Optimization-AI/SogCLR.git
```

### Training  
Below is an example for self-supervised pre-training of a ResNet-50 model on ImageNet on a 4-GPU server. By default, we use sqrt learning rate scaling, i.e., $\text{LearningRate}=0.075\times\sqrt{\text{BatchSize}}$, [LARS](https://arxiv.org/abs/1708.03888) optimizer and a weight decay of 1e-6. For temperature parameter $\tau$, we use a fixed value $0.1$ from [SimCLR](https://arxiv.org/pdf/2002.05709.pdf). For GCL, gamma (γ in the paper) is an additional parameter for maintaining moving average estimator, the default value is $0.9$, however, it is recommended to tune this parameter in the range of $[0.1\sim 0.99]$ for better performance. 


**ImageNet1K**

We use a batch size of 256 and pretrain ResNet-50 for 800 epochs. You can also increase the number of workers to accelerate the training speed. 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --lr=.075 --epochs=800 --batch-size=256 \
  --learning-rate-scaling=sqrt \
  --loss_type dcl \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10001' \
  --data_name imagenet1000 \
  --data /your-data-path/imagenet1000/ \
  --save_dir /your-data-path/saved_models/ \
  --print-freq 1000
```


**ImageNet100**

We also used a small version of ImageNet1K for experiments, i.e., ImageNet-100 is a subset with random selected 100 classes from original 1000 classes. To contrust the dataset, please follow these steps:
* Download the train and validation datasets from [ImageNet1K](https://image-net.org/challenges/LSVRC/2012/) website
* Run this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to create/move all validation images to each category (class) folder
* Copy images from [train/val.txt](https://github.com/Optimization-AI/SogCLR/blob/main/dataset/ImageNet-S/train.txt) to generate ImageNet-100

We use a batch size of 256 and pretrain ResNet-50 for 400 epochs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --lr=.075 --epochs=400 --batch-size=256 \
  --learning-rate-scaling=sqrt \
  --loss_type dcl \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10001' \
  --data_name imagenet100 \
  --data /your-data-path/imagenet100/ \
  --save_dir /your-data-path/saved_models/ \
  --print-freq 1000
```

### Linear evaluation
By default, we use momentum-SGD without weight decay and a batch size of 1024 for linear evaluation on on frozen features/weights. In this stage, it runs 90 epochs for re-training the classifiers.

**ImageNet**

```bash
python lincls.py \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 \
  --pretrained /your-data-path/checkpoint_0799.pth.tar
  --data_name imagenet1000 \
  --data /your-data-path/imagenet1000/ \
  --save_dir /your-data-path/saved_models/ \
```

## Benchmarks

The following results are linear evaluation results on **ImageNet1K** validation set:

| Method | BatchSize |Epoch | Linear eval. |
|:----------:|:--------:|:--------:|:--------:|
| SimCLR (TF[^2]) | 256   |   800 | 66.5 |
| SogCLR (PT[^1]) | 256   |   800 | 69.0 |
| SogCLR (TF[^2]) | 256   |   800 | 69.3 |

*SogCLR (PT[^1]): pre-trained ResNet-50 checkpoint & linear evaluation training log can be downloaded here: [[checkpoint_0799.pth.tar](https://drive.google.com/file/d/1baWWT6Xf9ylLHimWXZuhvdiKUkkMLB0_/view?usp=sharing) | [linear_eval.txt](https://drive.google.com/file/d/1O2N90Ffk0Oz6dXek_MhEVgXzszaogfvy/view?usp=sharing)]


The following results are linear evaluation results on **ImageNet-100** validation set:

| Method | BatchSize |Epoch | Linear eval. |
|:----------:|:--------:|:--------:|:--------:|
| SimCLR (TF[^2]) | 256   |   400 | 76.1 |
| SogCLR (PT[^1]) | 256   |   400 | 80.0 |
| SogCLR (TF[^2]) | 256   |   400 | 78.7 |

[^1]: PyTorch (PT) is based on [MoCo's](https://github.com/facebookresearch/moco) codebase.
[^2]: Tensorflow (TF) is based on [SimCLR's](https://github.com/google-research/simclr/tree/master/tf2) codebase.

The following results are the comparsion of SogCLR and SimCLR using different batch sizes for 800-epoch pretraining on ImageNet-1K. 

<img src="https://raw.githubusercontent.com/Optimization-AI/SogCLR/Tensorflow/imgs/sogclr_batch_size.png" width="383" height="238">


### Reference
If you find this tutorial helpful, please cite our paper:
```
@inproceedings{yuan2022provable,
  title={Provable stochastic optimization for global contrastive learning: Small batch does not harm performance},
  author={Yuan, Zhuoning and Wu, Yuexin and Qiu, Zi-Hao and Du, Xianzhi and Zhang, Lijun and Zhou, Denny and Yang, Tianbao},
  booktitle={International Conference on Machine Learning},
  pages={25760--25782},
  year={2022},
  organization={PMLR}
}
```

