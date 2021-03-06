# SogCLR [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/pdf/2202.12387.pdf)

This is the official implementation of the paper "**Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance**". Our algorithm can train self-supervised models with smaller batch sizes. The code can be run on TPUs or GPUs. 

Requirements
---
```
tensorflow==2.7.0
tensorflow-datasets
```

Datasets
---
**ImageNet-S** is a subset of **ImageNet-1K** with random selected 100 classes from original 1000 classes. You could follow the instruction [here](https://github.com/Optimization-AI/sogclr/tree/main/dataset) to convert dataset to tfrecord. To run SogCLR, we need image IDs to track the running statistics and each tfrecord should contain the following features: **image**, **label**, **id**, such as:

```Python
features=tfds.features.FeaturesDict({
            'image/encoded': tfds.features.Image(encoding_format='jpeg'),
            'image/class/label': tfds.features.ClassLabel(names_file=names_file),
            'image/ID': tf.int64,  # eg: 0,1,2,3,4,...,N
        })
```

Copy the provided `/code/imagenet.py` to your local directory under `tensorflow_datasets`:
```bash
cp imagenet.py  /usr/local/lib/python3.7/dist-packages/tensorflow_datasets/image_classification/imagenet.py 
```
Specify the `num_classes` and `data_dir`:
- ImageNet-S: --num_classes=100 --data_dir=gs://\<path-to-tensorflow-dataset\>
- ImageNet-1K: --num_classes=1000 --data_dir=gs://\<path-to-tensorflow-dataset\>


Pretraining
---
To pretrain the ResNet50 on ImageNet-1K using **SogCLR** + **Dynamic Contrastive Loss** with TPUs, you could set `GCL_mode=True` and `gamma=0.9` and then run the following command:
```bash
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=800 --temperature=0.1 \
  --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-6 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation --num_classes=1000 \
  --num_proj_layers=2 \
  --DCL_mode=True --gamma=0.9  \
  --data_dir=gs://<path-to-tensorflow-dataset> \
  --model_dir=gs://<path-to-store-checkpoints> \
  --use_tpu=True
```
For baseline, you could set `BI_mode=False` to use SimCLR. To use a 3-layer MLP projection head, you could set `num_proj_layers=3`. To use larger ResNet models, you could set `width_multiplier=1,2,4` and/or `resnet_depth=50,101,152`. To use GPU, you could set `use_tpu=False`. 

Linear Evaluation
---
By default `lineareval_while_pretraining=True`, it will train the linear classifier with a `stop_gradient` operator during pretraining, which is simlar to perform linear evaluation after pretraining [[Ref]](https://github.com/google-research/simclr/issues/151). An example of command line for linear evaluation is as follow:
```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --num_proj_layers=0 --ft_proj_selector=-1 \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 \
  --learning_rate_scaling=linear --weight_decay=0 \
  --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=gs://<path-to-tensorflow-dataset> \
  --model_dir=gs://<path-to-store-checkpoints> \
  --checkpoint=gs://<path-to-store-checkpoint>/ckpt-xxxx' \
  --use_tpu=True 
```


Citation
---------
If you find this repo helpful, please cite the following paper:

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

Reference
---
- https://github.com/google-research/simclr/tree/master/tf2
- https://cloud.google.com/tpu/docs/tutorials/mnist

            
