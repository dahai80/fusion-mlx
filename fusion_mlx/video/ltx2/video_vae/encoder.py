import mlx.core as mx

from .video_vae import VideoEncoder


def encode_image(
    image: mx.array,
    encoder: VideoEncoder,
) -> mx.array:
    if image.ndim == 3:
        image = mx.expand_dims(image, axis=0)

    image = mx.transpose(image, (0, 3, 1, 2))

    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2.0 - 1.0

    image = mx.expand_dims(image, axis=2)

    latent = encoder(image)

    return latent
