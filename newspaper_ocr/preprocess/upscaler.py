from aura_sr import AuraSR
from PIL import Image

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")


def upscale_image(image: Image.Image, alpha: float = 0.3):
    """Upscales a given image using a pre-trained AuraSR model and blends the upscaled image with the original to prevent artifacts.

    Args:
        image (PIL.Image.Image): The image to be upscaled.
        alpha (float): The blending factor for merging the original and upscaled images.

    Returns:
        PIL.Image.Image: The blended image with improved resolution.
    """
    try:
        upscaled_image = aura_sr.upscale_4x_overlapped(image)
    except Exception as e:
        print(f"Error upscaling image: {e}")
        return image
    resized_image = image.resize(upscaled_image.size, Image.LANCZOS)  # Resize the original image to match the size of upscaled_image
    mixed_image = Image.blend(resized_image, upscaled_image, alpha=alpha)
    return mixed_image.resize(image.size, Image.LANCZOS)
