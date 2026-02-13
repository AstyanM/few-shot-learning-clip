"""
Adaptive contrast enhancement for improved feature discriminability.
"""

import cv2
import numpy as np
import torch


def apply_adaptive_contrast_enhancement(image_tensor):
    """Apply adaptive contrast enhancement via HSV saturation boosting.

    Increases colour saturation to improve feature discriminability in the
    CLIP visual encoder's representation space.

    Args:
        image_tensor: Image tensor of shape ``[C, H, W]`` or ``[B, C, H, W]``
            normalised to approximately ``[-1, 1]``.

    Returns:
        Enhanced tensor with the same shape and device as the input.
    """
    is_batch = len(image_tensor.shape) == 4
    if not is_batch:
        image_tensor = image_tensor.unsqueeze(0)

    enhanced_batch = []

    for img in image_tensor:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)

        enhanced_img = _apply_transformation(img_np)

        enhanced_img = enhanced_img.astype(np.float32) / 255.0
        enhanced_img = (enhanced_img - 0.5) / 0.5
        enhanced_tensor = torch.from_numpy(enhanced_img).permute(2, 0, 1)
        enhanced_batch.append(enhanced_tensor)

    enhanced_batch = torch.stack(enhanced_batch).to(image_tensor.device)

    if not is_batch:
        enhanced_batch = enhanced_batch.squeeze(0)

    return enhanced_batch


def _apply_transformation(img_np, factor=1.2):
    """Boost saturation in HSV colour space.

    Args:
        img_np: RGB uint8 image array ``[H, W, 3]``.
        factor: Saturation multiplication factor.

    Returns:
        Enhanced RGB uint8 image array.
    """
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * factor, 0, 255)
    img_boosted = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_boosted
