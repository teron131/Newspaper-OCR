import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for line detection."""
    # Convert BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def detect_lines(image: np.ndarray, thresh: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Detect lines and calculate angles."""
    image_debug = image.copy()
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)
    angles = []

    if lines is None:
        return angles, image_debug

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if x2 != x1:  # Avoid division by zero
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)

    return angles, image_debug


def calculate_rotation_matrix(image: np.ndarray, angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Calculate rotation matrix and new dimensions."""
    # We assume it is always slightly skewed, so it can be adjusted by either clockwise or anti-clockwise rotation less than 90 degrees
    # Postive angle for anti-clockwise rotation
    # Negative angle for clockwise rotation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions based on rotation matrix
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return M, (new_w, new_h)


def deskew_image(image: Image.Image, plot: bool = False) -> Image.Image:
    """Deskew an image by detecting and correcting its skew angle.

    Args:
        image_path (str): Path to the input image file

    Returns:
        Image.Image: Deskewed PIL Image
    """
    # Convert PIL Image to OpenCV format (RGB to BGR)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    thresh = preprocess_image(image)

    # Detect lines and calculate angles
    angles, debug_image = detect_lines(image, thresh)

    if not angles:
        return image

    # Calculate most common angle after filtering outliers
    angles = np.array(angles)
    angles = angles[np.abs(angles - np.median(angles)) < 2 * np.std(angles)]
    hist, bins = np.histogram(angles, bins=np.arange(min(angles), max(angles) + 1, 1))
    angle = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2

    # Calculate rotation matrix
    M, (new_w, new_h) = calculate_rotation_matrix(image, angle)

    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if plot:
        # Create a figure with four subplots side by side
        plt.figure(figsize=(16, 4))

        # Plot original image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Target Image")
        plt.axis("off")

        # Plot debug image with detected lines
        plt.subplot(142)
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Lines")
        plt.axis("off")

        # Plot angle distribution
        plt.subplot(143)
        plt.hist(angles, bins=bins, edgecolor="black")
        plt.title("Distribution of Detected Angles")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Plot deskewed result
        plt.subplot(144)
        plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        plt.title("Deskewed Image")
        plt.axis("off")

        display(plt.gcf())
        plt.close()

    # Convert BGR back to RGB for PIL Image
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
