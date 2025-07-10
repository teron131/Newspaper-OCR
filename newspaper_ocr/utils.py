from pathlib import Path

import cv2
import numpy as np
from google.cloud import documentai
from IPython.display import display
from PIL import Image


def get_mime_type(file_path: str) -> str:
    """Get MIME type from file extension, specifically for the inputs of Google Document AI.
    Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    """
    extension = Path(file_path).suffix.lower().lstrip(".")
    mime_types = {
        "pdf": "application/pdf",
        "gif": "image/gif",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "html": "text/html",
    }
    return mime_types.get(extension)


def display_resize(image: Image.Image, size: tuple[int, int] = (1000, 1000)) -> None:
    """For ipynb, resize the image to 1000x1000 and display it."""
    image.thumbnail(size, Image.Resampling.LANCZOS)
    display(image)


def draw_boxes(result: documentai.Document) -> Image.Image:
    """Parse the Document result to draw boxes on the image.

    Args:
        result: The Document result to draw boxes on.

    Returns:
        Image.Image: The image with boxes drawn on it.
    """
    page = result.pages[0]
    img = cv2.imdecode(np.frombuffer(page.image.content, np.uint8), cv2.IMREAD_COLOR)

    for symbol in page.symbols:
        points = [[v.x, v.y] for v in symbol.layout.bounding_poly.vertices]
        color = np.array([0, 255, 0]).tolist()  # Green
        cv2.polylines(img, [np.array(points, np.int32)], True, color, 2)

    # Convert to PIL and display
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def erase_boxes(result: documentai.Document) -> Image.Image:
    """Parse the Document result to erase boxes from the image.

    Args:
        result: The Document result to erase boxes from.

    Returns:
        Image.Image: The image with boxes erased.
    """
    page = result.pages[0]
    img = cv2.imdecode(np.frombuffer(page.image.content, np.uint8), cv2.IMREAD_COLOR)

    for block in page.blocks:
        # Convert points to numpy array of int32
        points = np.array([[v.x, v.y] for v in block.layout.bounding_poly.vertices], dtype=np.int32)
        # Fill polygon with white
        cv2.fillPoly(img, [points], (255, 255, 255))

    # Convert to PIL and display
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
