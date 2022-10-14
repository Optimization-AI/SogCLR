# SogCLR PyTorch Implementation

In this repo, you will learn how to train a self-supervised model by using [Global Contrastive Loss](https://arxiv.org/abs/2202.12387) (GCL) on CIFAR10/CIFAR100. The original GCL was implementated in Tensorflow and run in TPUs. This repo re-implements GCL in PyTorch and GPUs based on [moco's](https://github.com/facebookresearch/moco) codebase. We recommend users to run this notebook on a GPU-enabled environment, e.g., [Google Colab](https://colab.research.google.com/). 


## Installation

#### git clone
```bash
git clone -b cifar https://github.com/Optimization-AI/SogCLR.git
```

### Training  
Below is an example for self-supervised pre-training of a ResNet-50 model on CIFAR10 on a single GPU. The first time you run the scripts, datasets will be automatically downloaded to `/data/`. By default, we use linear learning rate scaling, e.g., $\text{LearningRate}=1.0\times\text{BatchSize}/256$, [LARS](https://arxiv.org/abs/1708.03888) optimizer and a weight decay of 1e-4. For temperature parameter $\tau$, we use a fixed value of 0.3. For DCL, gamma (γ in the paper) is an additional parameter for maintaining moving average estimator, the default value is 0.9. By default, `CIFAR10` is used for experiments. To pretrain on CIFAR100, you can set `--data_name cifar100`. In this repo, only `CIFAR10/CIFAR100` is supported, however, you can modify the dataloader to support other datasets.


**CIFAR**

We use batch size of 64 and train 400 epochs for pretraining. You can also increase the number of workers to accelerate the training speed.

```bash
python train.py \
  --lr=1.0 --learning-rate-scaling=sqrt \
  --epochs=400 --batch-size=64 \
  --loss_type dcl \
  --gamma 0.9 \
  --workers 32 \
  --wd=1e-4 \
  --data_name cifar10 \
  --save_dir ./saved_models/ \
  --print-freq 1000
```

### Linear evaluation
By default, we use momentum-SGD without weight decay and a batch size of 1024 for linear evaluation on on frozen features/weights. In this stage, it runs 90 epochs for training.

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

