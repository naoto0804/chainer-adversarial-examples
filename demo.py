#!/usr/bin/env python3

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import cupy
import numpy as np

from attacks import fgsm
from attacks import tgsm
from mlp import MLP

N_gen = 5
img_size = (28, 28)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
parser.add_argument('--model', default='model/mlp_iter_12000',
                    help='path of already trained model')
args = parser.parse_args()

print('Using gpu ' + str(args.gpu))


def visualize(adv_images, prob, img_size, filename):
    n_images = adv_images.shape[0]
    fig = plt.figure(figsize=(n_images, 1.8))
    gs = gridspec.GridSpec(1, n_images, wspace=0.1, hspace=0.1)

    label = np.argmax(prob, axis=1)
    p = np.max(prob, axis=1)
    for i in range(n_images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(adv_images[i].reshape(img_size), cmap='gray',
                  interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(label[i], p[i]), fontsize=12)

    gs.tight_layout(fig)
    plt.savefig(filename)


def sample(dataset, n_samples):
    images, _ = test_mnist[np.random.choice(len(dataset), n_samples)]
    images = chainer.cuda.to_gpu(images, args.gpu)
    return images


# Setup model, dataset
model = MLP(args.unit, 10)
chainer.cuda.get_device_from_id(args.gpu).use()
model.to_gpu()
xp = chainer.cuda.get_array_module(model)
chainer.serializers.load_npz(args.model, model)
_, test_mnist = chainer.datasets.get_mnist()

# Fast Gradient Sign Method (simple)
images = sample(test_mnist, N_gen)
adv_images = fgsm(model, images, eps=0.2)
prob = F.softmax(model(adv_images), axis=1).data
visualize(cupy.asnumpy(adv_images), cupy.asnumpy(prob), img_size, 'fgsm.png')

# Fast Gradient Sign Method (iterative)
images = sample(test_mnist, N_gen)
adv_images = fgsm(model, images, eps=0.01, iterations=20)
prob = F.softmax(model(adv_images), axis=1).data
visualize(cupy.asnumpy(adv_images), cupy.asnumpy(prob), img_size,
          'fgsm_iterative.png')

# Target class Gradient Sign Method (least-likely)
images = sample(test_mnist, N_gen)
adv_images = tgsm(model, images, eps=0.15)
prob = F.softmax(model(adv_images), axis=1).data
visualize(cupy.asnumpy(adv_images), cupy.asnumpy(prob), img_size,
          'tgsm.png')

# Target class Gradient Sign Method (target assigned)
images = sample(test_mnist, N_gen)
adv_images = tgsm(model, images, target=8, eps=0.3)
prob = F.softmax(model(adv_images), axis=1).data
visualize(cupy.asnumpy(adv_images), cupy.asnumpy(prob), img_size,
          'tgsm_target_assigned.png')
