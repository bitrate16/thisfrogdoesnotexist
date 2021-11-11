# thisfrogdoesnotexist.com

## Background

The main goal of this project is to implement GANs to generate images of frogs and toads despite to variations of colors and shapes.

Current implementation is based on DCGAN from [this article](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

Additionally, added support for processing 128 x 128 images, Dropout layers for Generator network and noise input for real image samples.

# Dataset

Real frog images are from [jonshamir repository](https://github.com/jonshamir/frog-dataset/), additional input augmentations (Rotation, Mirror, HSV shift) are defined in [transform.py](transform.py) file.

## Pretrained

Pretrained models are listed in [models README](models/README.md).

## Notebooks

Notebooks are located in [notebooks folder](notebooks).

## Train manually

[Train script](train.py) is located in the root directory and suppots following list of arguments:

```

```

## Generate

[Generation script](geenrate.py) is located in the root directory and suppots following list of arguments:

```

```

## Future work

* train.py & generate.py scripts
* Research for best combination of hyperparameters for Generator & Discriminator
* Figure out with generation for 128 x 128 images
* Augmentations using Pytorch random rotation & mirror
* Try to use STYLEGAN (Currently unavailable due unavailability of GPU)
* Keep frog wet
