"""Training support for rectified source images, never an admission mask."""
import numpy as np


def reconstruction_loss_support(map_x, map_y, *, source_size, excluded_rectangles=(),
                                ssim_window_size=11):
    """Keep only SSIM windows whose every pixel has a genuine source observation.

    Evaluation must still follow its own frozen protocol. This mask only protects
    reconstruction updates from artificial remap borders and unmodeled actors.
    """
    import cv2

    x, y = np.asarray(map_x), np.asarray(map_y)
    if (x.ndim != 2 or x.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(y).all()):
        raise ValueError("Finite equally shaped rectification maps required")
    if (len(source_size) != 2 or any(type(n) is not int or n <= 0 for n in source_size)
            or type(ssim_window_size) is not int or ssim_window_size <= 0
            or ssim_window_size % 2 != 1):
        raise ValueError("Positive source size and odd SSIM window required")
    width, height = source_size
    support = cv2.remap(np.ones((height, width), np.float32), x.astype(np.float32),
                        y.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= .9999
    out_h, out_w = support.shape
    for rectangle in excluded_rectangles:
        if (len(rectangle) != 4 or any(type(n) is not int for n in rectangle)):
            raise ValueError("Integer XYXY exclusion rectangles required")
        x0, y0, x1, y1 = rectangle
        if not 0 <= x0 < x1 <= out_w or not 0 <= y0 < y1 <= out_h:
            raise ValueError("Exclusion rectangle is outside rectified image")
        support[y0:y1, x0:x1] = False
    return cv2.erode(support.astype(np.uint8),
                     np.ones((ssim_window_size, ssim_window_size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)[..., None]
