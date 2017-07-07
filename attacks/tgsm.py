import chainer
import chainer.functions as F


def tgsm(model, images, target=None, eps=0.01, iterations=1, clip_min=0.,
         clip_max=1.):
    """ Computing adversarial images based on Target class Gradient Sign Method.

    Args:
        model (chainer.Link): Predictor network excluding softmax.
        images (numpy.ndarray or cupy.ndarray): Initial images.
        target (None or int or list of integers): Target class.
            If target is None, this implements least-likely class method.
            If target is int or list of integers, the original images are
                modified towards label target.
        eps (float): Attack step size.
        iterations (int): Number of attack iterations.
        clip_min (float): Minimum input component value.
        clip_max (float): Maximum input component value.

    Returns:
        adv_images (numpy.ndarray or cupy.ndarray):
            Generated adversarial images.

    Reference:
        Adversarial examples in the physical world,
        Kurakin et al., ICLR2017, https://arxiv.org/abs/1607.02533

    """
    n_batch = images.shape[0]
    adv_images = images
    xp = chainer.cuda.get_array_module(adv_images)
    if target is None:
        targets = F.argmin(model(images), axis=1)
    else:
        if isinstance(target, int):
            targets = xp.full(n_batch, target).astype(xp.int32)
        elif isinstance(target, list):
            assert (len(target) == n_batch)
            targets = xp.array(target).astype(xp.int32)
        else:
            raise NotImplementedError

    eps = -xp.abs(eps)

    for _ in range(iterations):
        adv_images = chainer.Variable(adv_images)
        loss = F.softmax_cross_entropy(model(adv_images), targets)
        loss.backward()
        adv_images = adv_images.data + eps * xp.sign(adv_images.grad)
        adv_images = xp.clip(adv_images, a_min=clip_min, a_max=clip_max)
    return adv_images.astype(xp.float32)
