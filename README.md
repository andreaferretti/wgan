# Wasserstein GAN with Gradient Penalty in Pytorch

This is a cleanup of [improved-wgan-pytorch](https://github.com/jalola/improved-wgan-pytorch),
which implements methods from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training)
to train Wasserstein GAN with Gradient Penalty.

## Prerequisites

This project uses PyTorch (any recent version will do, we are using 1.2),
and [LMDB](https://lmdb.readthedocs.io/en/release/) (the latter is needed to
load data in [LSUN](https://github.com/fyu/lsun) format).

Optionally, it can log events using [tensorboardX](https://github.com/lanpa/tensorboardX)
that can be displayed using [TensorBoard](https://www.tensorflow.org/tensorboard).

For instance, you can create an [Anaconda](https://anaconda.org) environment
to run this project using something like

```
conda create -n wgan python=3.6
source activate wgan
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install lmdb
pip install tensorboardX # optional
```

## Usage

If you have a folder of images, structured like this

```
.
└── images
    ├── class1
    │   ├── img001.png
    │   ├── img002.png
    │   └── ...
    ├── class2
    │   └── ...
    └── ...
```

use `--dataset raw --data-dir images`, while for the LSUN dataset

```
.
└── LSUN
    ├── bedroom_train_lmdb
    │   ├── data.mdb
    │   └── lock.mdb
    ├── bedroom_val_lmdb
    │   ├── data.mdb
    │   └── lock.mdb
    ├── bridge_train_lmdb
    │   ├── data.mdb
    │   └── lock.mdb
    ├── bridge_val_lmdb
    │   ├── data.mdb
    │   └── lock.mdb
    └── ...
```

use something like `--dataset lsun --data-dir LSUN --image-class bedroom`.

All options are described in the help below

```
usage: train.py [-h] --data-dir DATA_DIR --output-dir OUTPUT_DIR --model-dir
                MODEL_DIR [--image-class IMAGE_CLASS] [--dataset {raw,lsun}]
                [--restore] [--image-size IMAGE_SIZE]
                [--state-size STATE_SIZE] [--batch-size BATCH_SIZE]
                [--epochs EPOCHS]
                [--generator-iterations GENERATOR_ITERATIONS]
                [--critic-iterations CRITIC_ITERATIONS]
                [--sample-every SAMPLE_EVERY]
                [--gradient-penalty GRADIENT_PENALTY]
                [--generator-lr GENERATOR_LR] [--critic-lr CRITIC_LR]

Wasserstein GAN

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   directory with the images
  --output-dir OUTPUT_DIR
                        directory where to store the generated images
  --model-dir MODEL_DIR
                        directory where to store the models
  --image-class IMAGE_CLASS
                        class to train on, only for LSUN
  --dataset {raw,lsun}  format of the dataset
  --restore             restart training from the saved models
  --image-size IMAGE_SIZE
                        image dimension
  --state-size STATE_SIZE
                        state size
  --batch-size BATCH_SIZE
                        batch size
  --epochs EPOCHS       number of epochs
  --generator-iterations GENERATOR_ITERATIONS
                        number of iterations for the generator
  --critic-iterations CRITIC_ITERATIONS
                        number of iterations for the critic
  --sample-every SAMPLE_EVERY
                        how often to sample images
  --gradient-penalty GRADIENT_PENALTY
                        gradient penalty
  --generator-lr GENERATOR_LR
                        learning rate for the generator
  --critic-lr CRITIC_LR
                        learning rate for the critic
```

## TensorboardX

Results such as costs, generated images for tensorboard will be written to `./runs` folder.

To display the results to tensorboard, run: `tensorboard --logdir runs`

## Acknowledgements

* [github.com/jalola/improved-wgan-pytorch](https://github.com/jalola/improved-wgan-pytorch)
* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)

## License

This project is licensed under [Apache2](https://opensource.org/licenses/Apache-2.0).
It is based on [improved-wgan-pytorch](https://github.com/jalola/improved-wgan-pytorch),
which is itself licensed under [MIT](https://opensource.org/licenses/MIT).