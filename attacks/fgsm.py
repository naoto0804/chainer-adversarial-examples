import chainer
import chainer.functions as F


def fgsm(model, images, eps=0.01, iterations=1, clip_min=0., clip_max=1.):
    """ Computing adversarial images based on Fast Gradient Sign Method or
        Basic Iterative Method.

    Args:
        model (chainer.Link): Predictor network excluding softmax.
        images (numpy.ndarray or cupy.ndarray): Initial images.
        eps (float): Attack step size.
        iterations (int): Number of attack iterations.
            If iterations = 1, this implements Fast Gradient Sign Method.
            If iterations > 1, this implements Basic Iterative Method.
        clip_min (float): Minimum input component value.
        clip_max (float): Maximum input component value.

    Returns:
        adv_images (numpy.ndarray or cupy.ndarray):
            Generated adversarial images.

    Reference:
        (Fast Gradient Sign Method)
        Explaining and Harnessing Adversarial Examples,
        Goodfellow et al., CoRR2014, https://arxiv.org/abs/1412.6572

        (Basic Iterative Method)
        Adversarial examples in the physical world,
        Kurakin et al., ICLR2017, https://arxiv.org/abs/1607.02533

    """

    adv_images = images
    xp = chainer.cuda.get_array_module(adv_images)
    targets = F.argmax(model(images), axis=1)
    eps = xp.abs(eps)

    for _ in range(iterations):
        adv_images = chainer.Variable(adv_images)
        loss = F.softmax_cross_entropy(model(adv_images), targets)
        loss.backward()
        adv_images = adv_images.data + eps * xp.sign(adv_images.grad)
        adv_images = xp.clip(adv_images, clip_min, clip_max).astype(xp.float32)
    return adv_images.astype(xp.float32)
