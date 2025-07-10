import base64
import io
from dataclasses import dataclass

from PIL import Image

from .clarity import adjust_contrast, adjust_whitebalance, denoise_image
from .deskew import deskew_image


@dataclass
class PreprocessOptions:
    """Options for image preprocess to improve OCR quality.

    Args:
        apply_deskew (bool): Whether to apply deskewing to correct image skew
        apply_whitebalance (bool): Whether to apply white balance adjustment
        apply_contrast (bool): Whether to apply contrast enhancement
        apply_denoise (bool): Whether to apply denoising
        wb_percentile (float): Percentile value to use for reference white (0-100)
        scale_factor (float): Scaling factor for whitebalance adjustment strength (0.0-2.0)
        auto (bool): Whether to use automatic whitebalance
        contrast (float): Contrast control (1.0-3.0)
        brightness (int): Brightness control (0-100)
    """

    apply_deskew: bool = True
    apply_whitebalance: bool = True
    apply_contrast: bool = False
    apply_denoise: bool = False
    wb_percentile: float = 99.0
    scale_factor: float = 1.0
    auto: bool = False
    contrast: float = 1.15
    brightness: int = 0


def load_preprocess_image(
    image_path: str,
    preprocess: bool = True,
    preprocess_options: PreprocessOptions = PreprocessOptions(),
) -> Image.Image:
    """Preprocess an image from a file path to specific format.

    Args:
        image_path (str): Path to the image file
        preprocess (bool): Whether to preprocess the image
        options (PreprocessOptions): Options for image preprocess

    Returns:
        Image.Image: Processed image
    """
    image = Image.open(image_path)

    if preprocess:

        if preprocess_options.apply_deskew:
            image = deskew_image(image)

        if preprocess_options.apply_whitebalance:
            image = adjust_whitebalance(
                image,
                wb_percentile=preprocess_options.wb_percentile,
                scale_factor=preprocess_options.scale_factor,
                auto=preprocess_options.auto,
            )

        if preprocess_options.apply_contrast:
            image = adjust_contrast(
                image,
                contrast=preprocess_options.contrast,
                brightness=preprocess_options.brightness,
            )

        if preprocess_options.apply_denoise:
            image = denoise_image(image)

    return image


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert an image to bytes.

    Args:
        image: The image to convert to bytes.

    Returns:
        bytes: The bytes of the image.
    """
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    return image_bytes


def image_to_base64(image: Image.Image) -> str:
    """Convert an image to base64.

    Args:
        image: The image to convert to base64.

    Returns:
        str: The base64 of the image.
    """
    image_bytes = image_to_bytes(image)
    return base64.b64encode(image_bytes).decode("utf-8")
