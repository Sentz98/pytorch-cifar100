import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

class CorruptTransform:
    def __init__(self, method=None, severity_range=(1, 3), same_corruption=False):
        self.ic=ImageCorruptor()
        self.method=method
        self.severity_range=severity_range
        self.same_corruption=same_corruption
    

    def __call__(self, img):
        return self.ic.corrupt_batch(img.unsqueeze(0), self.method, self.severity_range, self.same_corruption)[0]  # Mock batch of 1



class ImageCorruptor:
    """
    A class that provides various image corruption methods to demonstrate
    the effectiveness of dynamic quantization versus static quantization.
    Works directly with PyTorch tensors.
    """

    def __init__(self, seed=1234):
        """
        Initialize the ImageCorruptor with an optional random seed.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.available_corruptions = [
            "noise",
            "blur",
            "pixelate",
            "quantize",
            "color_shift",
            "brightness",
            "contrast",
            # "distortion",
            "combined",
        ]

    def corrupt(self, images, method, severity=1):
        """
        Apply a corruption method to batched PyTorch tensors with a given severity.

        Args:
            images: PyTorch tensor of shape [B, C, H, W] in range [0, 1]
            method (str): Corruption method to apply
            severity (int): Severity level from 1 (mild) to 5 (severe)

        Returns:
            torch.Tensor: Corrupted images with the same shape as input
        """
        # Ensure images are in the correct shape [B, C, H, W]
        if len(images.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], got shape {images.shape}"
            )

        # Clamp severity to valid range
        severity = max(1, min(5, severity))

        # Apply the requested corruption
        if method == "noise":
            return self._add_noise(images, severity)
        elif method == "blur":
            return self._add_blur(images, severity)
        elif method == "pixelate":
            return self._pixelate(images, severity)
        elif method == "quantize":
            return self._color_quantize(images, severity)
        elif method == "color_shift":
            return self._color_shift(images, severity)
        elif method == "brightness":
            return self._adjust_brightness(images, severity)
        elif method == "contrast":
            return self._adjust_contrast(images, severity)
        elif method == "distortion":
            return self._add_distortion(images, severity)
        elif method == "combined":
            return self._combined_corruptions(images, severity)
        else:
            raise ValueError(
                f"Unknown corruption method: {method}. Available methods: {self.available_corruptions}"
            )

    def corrupt_batch(
        self, images, methods=None, severity_range=(1, 3), same_corruption=False
    ):
        """
        Apply random corruptions to a batch of images.

        Args:
            images: PyTorch tensor of shape [B, C, H, W]
            methods (list, optional): List of methods to choose from. Defaults to all available.
            severity_range (tuple): Range of severity levels to choose from.
            same_corruption (bool): Whether to apply the same corruption to all images in batch.

        Returns:
            torch.Tensor: Corrupted images with the same shape as input
        """
        if methods is None:
            methods = self.available_corruptions

        min_severity, max_severity = severity_range
        batch_size = images.shape[0]

        if same_corruption:
            # Apply the same corruption to all images in the batch
            method = random.choice(methods)
            severity = random.randint(min_severity, max_severity)
            return self.corrupt(images, method, severity)
        else:
            # Apply different corruptions to each image in the batch
            corrupted_images = []
            for i in range(batch_size):
                method = random.choice(methods)
                severity = random.randint(min_severity, max_severity)
                # Extract single image, apply corruption, and keep dimensions
                single_img = images[i : i + 1]
                corrupted = self.corrupt(single_img, method, severity)
                corrupted_images.append(corrupted)

            # Concatenate back into a batch
            return torch.cat(corrupted_images, dim=0)

    def _add_noise(self, images, severity):
        """Add Gaussian noise to image tensors."""
        noise_level = severity * 0.1
        noise = torch.randn_like(images) * noise_level
        noisy_img = torch.clamp(images + noise, 0, 1)
        return noisy_img

    def _add_blur(self, images, severity):
        """Apply Gaussian blur to image tensors."""
        # Kernel size increases with severity
        kernel_size = 2 * severity + 1
        sigma = 0.5 * severity

        # Apply Gaussian blur
        return transforms.GaussianBlur(kernel_size, sigma=sigma)(images)

    def _pixelate(self, images, severity):
        """Pixelate the images by downsampling and upsampling."""
        batch_size, channels, height, width = images.shape

        # Downsampling factor increases with severity
        factor = 6 - severity  # Higher severity = smaller factor = more pixelation

        # Downsample
        small_h, small_w = height // factor, width // factor
        downsampled = F.interpolate(images, size=(small_h, small_w), mode="nearest")

        # Upsample back to original size
        pixelated = F.interpolate(downsampled, size=(height, width), mode="nearest")

        return pixelated

    def _color_quantize(self, images, severity):
        """Reduce the number of colors in the images."""
        # Calculate number of quantization levels based on severity
        levels = 256 // (2**severity)

        # Apply quantization
        factor = 1.0 / levels
        quantized = torch.floor(images / factor) * factor

        return torch.clamp(quantized, 0, 1)

    def _color_shift(self, images, severity):
        """Shift the color channels of the images."""
        batch_size, channels, height, width = images.shape

        # Convert to numpy for easier manipulation
        images_np = images.detach().cpu().numpy()
        shifted_images = np.zeros_like(images_np)

        for b in range(batch_size):
            # Shift each channel separately
            for c in range(channels):
                shift_amount = int(severity * 10)
                if shift_amount > 0:
                    dx = random.randint(-shift_amount, shift_amount)
                    dy = random.randint(-shift_amount, shift_amount)

                    # Apply shift using OpenCV for efficiency
                    channel = images_np[b, c]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine(
                        channel, M, (width, height), borderMode=cv2.BORDER_REFLECT
                    )
                    shifted_images[b, c] = shifted
                else:
                    shifted_images[b, c] = images_np[b, c]

        # Convert back to PyTorch tensor
        return torch.from_numpy(shifted_images).to(images.device)

    def _adjust_brightness(self, images, severity):
        """Adjust the brightness of the images."""
        factor = 1.0 + (severity * 0.2)
        return torch.clamp(images * factor, 0, 1)

    def _adjust_contrast(self, images, severity):
        """Adjust the contrast of the images."""
        factor = 1.0 + (severity * 0.3)

        # Calculate per-channel means for each image in batch
        means = images.mean(dim=[2, 3], keepdim=True)

        # Apply contrast adjustment
        adjusted = factor * (images - means) + means
        return torch.clamp(adjusted, 0, 1)

    def _add_distortion(self, images, severity):
        """Add elastic distortion to the images."""
        batch_size, channels, height, width = images.shape

        # Convert to numpy for distortion
        images_np = images.detach().cpu().numpy()
        distorted_images = np.zeros_like(images_np)

        for b in range(batch_size):
            # Create distortion grid
            x, y = np.meshgrid(np.arange(width), np.arange(height))

            # Distortion strength based on severity
            strength = severity * 5

            # Create random displacement fields
            dx = strength * np.random.rand(height, width) * 2 - strength
            dy = strength * np.random.rand(height, width) * 2 - strength

            # Apply Gaussian filter to smooth the distortion
            dx = cv2.GaussianBlur(dx, (0, 0), severity * 2)
            dy = cv2.GaussianBlur(dy, (0, 0), severity * 2)

            # Distort the coordinates
            x_distorted = np.clip(x + dx, 0, width - 1).astype(np.float32)
            y_distorted = np.clip(y + dy, 0, height - 1).astype(np.float32)

            # Apply distortion to each channel
            for c in range(channels):
                distorted_images[b, c] = cv2.remap(
                    images_np[b, c],
                    x_distorted,
                    y_distorted,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

        # Convert back to PyTorch tensor
        return torch.from_numpy(distorted_images).to(images.device)

    def _combined_corruptions(self, images, severity):
        """Apply multiple random corruptions in sequence."""
        # Choose 2-3 random corruptions based on severity
        num_corruptions = min(severity + 1, 3)
        corruptions = random.sample(self.available_corruptions[:-1], num_corruptions)

        # Apply each corruption with reduced severity to avoid extreme results
        corrupted = images
        for method in corruptions:
            reduced_severity = max(1, severity - 1)
            corrupted = self.corrupt(corrupted, method, reduced_severity)

        return corrupted
