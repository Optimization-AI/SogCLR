# SogCLR PyTorch Implementation

In this repo, you will learn how to train a self-supervised model by optimizing [Global Contrastive Loss](https://arxiv.org/abs/2202.12387) (GCL) on CIFAR10/CIFAR100. The original GCL was implementated in Tensorflow and run in TPUs. This repo re-implements DCL in PyTorch and GPUs based on [moco's](https://github.com/facebookresearch/moco) codebase. We recommend users to run this example on a GPU-enabled environment, e.g., [Google Colab](https://colab.research.google.com/). 


## Installation

#### git clone
```bash
git clone -b cifar https://github.com/Optimization-AI/SogCLR.git
```

### Training  
Below are two examples for self-supervised pre-training of a ResNet-50 model on CIFAR10 on a single GPU. The first time you run the scripts, datasets will be automatically downloaded to `/data/`. By default, we use linear learning rate scaling, e.g., $\text{LearningRate}=1.0\times\text{BatchSize}/256$, [LARS](https://arxiv.org/abs/1708.03888) optimizer and a weight decay of 1e-4. For temperature parameter $\tau$, we use a fixed value of 0.5 recommended in [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) paper. For DCL, gamma (λ) is an additional parameter for maintaining moving average estimator, the default value is 0.9. In this version, only CIFAR10/CIFAR100 is supported. To pretrain on CIFAR100, you can set `--data_name cifar100`.

**CIFAR**

```bash
python train.py \
  --lr=1.0 --learning-rate-scaling=sqrt \
  --epochs=100 --batch-size=linear \
  --loss_type dcl \
  --gamma 0.9 \
  --workers 32 \
  --wd=1e-4 \
  --data_name cifar10 \
  --save_dir ./saved_models/ \
  --print-freq 1000
```


### Linear evaluation
By default, we use momentum-SGD without weight decay and a batch size of 1024 for linear classification on frozen features/weights. The training takes 90 epochs.

**CIFAR**
```bash
python lincls.py \
  --workers 32 \
  --pretrained /path-to-checkpoint/checkpoint_0099.pth.tar
  --data_name cifar10 \
  --save_dir ./saved_models/ \
```

## Benchmarks

The following results are linear evaluation accuracy on CIFAR10 testing dataset. All results are based on a batch size of 64 for 400-epoch pretraining.

| Method | BatchSize |Epoch | Linear eval. |
|:----------:|:--------:|:--------:|:--------:|
| SimCLR | 64   |   400 |  90.66    |
| SogCLR | 64   |   400 | 91.78  |



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

