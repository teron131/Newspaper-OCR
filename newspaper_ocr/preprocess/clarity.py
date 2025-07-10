import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


def check_text_clarity(image: Image.Image, plot: bool = False) -> tuple[Image.Image, dict]:
    """Quantifies text clarity by evaluating color contrast between text and background, and edge sharpness.

    1. Color contrast between potential text and background is evaluated by applying Otsu thresholding to the grayscale image.
    The normalized difference between the mean intensities of the resulting two classes (assumed text and background)
    quantifies the color separation.

    2. Text edge sharpness is measured by averaging the gradient magnitudes of strong edges, calculated using a Sobel filter.
    Sharper text produces more abrupt intensity changes, resulting in higher average gradient magnitudes.

    Args:
        image (Image.Image): Input image
        plot (bool): Whether to plot intermediate steps

    Returns:
        tuple[Image.Image, dict]: Original image and dictionary containing:
            - color_contrast: Normalized difference between text and background intensities.
            - edge_sharpness: Average gradient magnitude for strong edges.
    """

    # Convert image to grayscale
    gray = np.array(image.convert("L")).astype(np.float32)

    # ---- Metric 1: Color Contrast using Otsu Thresholding ----
    # Color contrast between potential text and background is evaluated by applying Otsu thresholding to the grayscale image. The normalized difference between the mean intensities of the resulting two classes (assumed text and background) quantifies the color separation.
    # Apply Gaussian blur to reduce noise before thresholding
    _, binary = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary > 0

    # Calculate contrast between text and background regions
    mean_background = np.mean(gray[binary])
    mean_text = np.mean(gray[~binary])
    color_contrast = abs(mean_background - mean_text) / 255.0

    # ---- Metric 2: Edge Sharpness ----
    # Compute gradients via Sobel filters
    # Text edge sharpness is measured by averaging the gradient magnitudes of strong edges, calculated using a Sobel filter. Sharper text produces more abrupt intensity changes, resulting in higher average gradient magnitudes.
    sobel_x = ndimage.sobel(gray, axis=0)
    sobel_y = ndimage.sobel(gray, axis=1)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Focus only on significant edges
    edge_threshold = 0.3 * np.max(grad_mag)
    edge_pixels = grad_mag[grad_mag > edge_threshold]
    if edge_pixels.size > 0:
        edge_sharpness = np.mean(edge_pixels)
    else:
        edge_sharpness = 0.0

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(np.array(image), cmap="gray")
        plt.title(f"Target Image")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(binary, cmap="gray")
        plt.title(f"Otsu's Threshold\nColor Contrast: {color_contrast:.2f}")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(grad_mag, cmap="gray")
        plt.title(f"Gradient Magnitude\nEdge Sharpness: {edge_sharpness:.2f}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    metrics = {
        "color_contrast": color_contrast,
        "edge_sharpness": edge_sharpness,
    }
    return image, metrics


def adjust_whitebalance(
    image: Image.Image,
    wb_percentile: float = 99,
    scale_factor: float = 1.0,
    auto: bool = False,
) -> Image.Image:
    """Adjust white balance of an image using a percentile-based approach.

    Args:
        image (Image.Image): Input image
        percentile (float): Percentile value to use for reference white (0-100)
        scale_factor (float): Scaling factor for adjustment strength (0.0-2.0)
        auto (bool): Whether to use automatic white balance

    Returns:
        Image.Image: White balanced image
    """
    if auto:
        return _auto_whitebalance(image)

    # Convert to numpy array and ensure RGB format
    img_np = np.array(image.convert("RGB"))
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # Calculate reference white values using percentile
    r_ref = np.percentile(r, wb_percentile)
    g_ref = np.percentile(g, wb_percentile)
    b_ref = np.percentile(b, wb_percentile)

    # Calculate scaling factors
    max_val = max(r_ref, g_ref, b_ref)
    r_scale = (max_val / r_ref - 1) * scale_factor + 1
    g_scale = (max_val / g_ref - 1) * scale_factor + 1
    b_scale = (max_val / b_ref - 1) * scale_factor + 1

    # Apply scaling while keeping values in valid range
    r_balanced = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    b_balanced = np.clip(b * b_scale, 0, 255).astype(np.uint8)

    # Combine channels
    balanced = np.dstack((r_balanced, g_balanced, b_balanced))
    return Image.fromarray(balanced)


def _auto_whitebalance(image: Image.Image) -> Image.Image:
    """Automatically adjust white balance of the image.
    https://docs.opencv.org/4.11.0/d9/d7a/classcv_1_1xphoto_1_1WhiteBalancer.html

    Args:
        image (Image.Image): Input image

    Returns:
        Image.Image: Enhanced image
    """
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    wb = cv2.xphoto.createSimpleWB()
    balanced = wb.balanceWhite(img_np)
    return Image.fromarray(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))


def adjust_contrast(
    image: Image.Image,
    contrast: float = 1.15,
    brightness: int = 0,
) -> Image.Image:
    """Enhance text clarity in scanned newspaper images by adjusting contrast
    using linear transformation.

    Args:
        image (Image.Image): Input image
        contrast (float): Contrast control (1.0-3.0)
        brightness (int): Brightness control (0-100)

    Returns:
        Image.Image: Enhanced image
    """
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    adjusted = cv2.convertScaleAbs(img_np, alpha=contrast, beta=brightness)
    adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(adjusted_rgb)


def denoise_image(image: Image.Image) -> Image.Image:
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(img_np)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
