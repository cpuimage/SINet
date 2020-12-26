# Unofficial Tensorflow 2 implementation of SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder

## Requirements

- python 3
- opencv-python
- numpy
- tensorflow >=2.3.1
- albumentations

## Model

SINet ([paper](https://arxiv.org/abs/1911.09099)) Accepted in WACV2020

Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Nicolas Monet, Jihwan Bang, Nojun Kwak

SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder

## Run example

- Preparing dataset

if you use custom dataset, fix the code and parms in dataloader.py(DatasetLoader's "Load" call).

- Train

```shell
nohup python3 distributed_train.py>train.log 2>&1 &
```

- View Log

```shell
tail -f train.log
```

- TensorBoard

```shell
tensorboard --logdir ./training/
```

## Citation

```shell 
@article{park2019sinet,
  title={SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Monet, Nicolas and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1911.09099},
  year={2019}
}
@article{heo2020adamp,
    title={Slowing Down the Weight Norm Increase in Momentum-based Optimizers},
    author={Heo, Byeongho and Chun, Sanghyuk and Oh, Seong Joon and Han, Dongyoon and Yun, Sangdoo and Uh, Youngjung and Ha, Jung-Woo},
    year={2020},
    journal={arXiv preprint arXiv:2006.08217},
}
```

## The meaning of this repository

Try to provide you with a relatively concise and standardized TensorFlow 2 custom training code example.

## Related discussion post

[TensorFlow 2.X，会是它走下神坛的开始？](https://www.jiqizhixin.com/articles/2020-12-25-4)
