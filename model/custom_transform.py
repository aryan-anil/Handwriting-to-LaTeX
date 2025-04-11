import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os

class ColorAugmentations:
    def __init__(self,
                 min_brightness=0.5,
                 max_brightness=1.5,
                 min_contrast=0.5,
                 max_contrast=1.5,
                 min_color_shift=-30,
                 max_color_shift=30,
                 min_gamma=0.5,
                 max_gamma=1.5,
                 min_kernel_size=3,
                 max_kernel_size=7,
                 min_sigma=0.1,
                 max_sigma=1.5):
        
        self.brightness_range = (min_brightness, max_brightness)
        self.contrast_range = (min_contrast, max_contrast)
        self.color_shift_range = (min_color_shift, max_color_shift)
        self.gamma_range = (min_gamma, max_gamma)
        self.kernel_size_range = (min_kernel_size, max_kernel_size)
        self.sigma_range = (min_sigma, max_sigma)
        
        # PyTorch color transformations
        self.color_jitter = T.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0,  # No saturation for B&W
            hue=0  # No hue for B&W
        )
        
    def random_channel_shift(self, image: np.ndarray) -> np.ndarray:
        """Randomly shift RGB channels while maintaining B&W appearance"""
        shifted = image.copy()
        for channel in range(3):  # RGB channels
            shift = random.uniform(self.color_shift_range[0], self.color_shift_range[1])
            shifted[:, :, channel] = np.clip(shifted[:, :, channel] + shift, 0, 255)
        return shifted
    
    def random_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply random gamma correction to maintain B&W but alter intensity"""
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def random_channel_multiplication(self, image: np.ndarray) -> np.ndarray:
        """Multiply each channel by a random factor"""
        multiplied = image.copy()
        for channel in range(3):
            factor = random.uniform(0.7, 1.3)
            multiplied[:, :, channel] = np.clip(multiplied[:, :, channel] * factor, 0, 255)
        return multiplied
    
    def pytorch_color_transforms(self, image: Image.Image) -> Image.Image:
        """Apply PyTorch's built-in color transformations"""
        return self.color_jitter(image)
    
    def sepia_tone(self, image: np.ndarray, intensity: float = None) -> np.ndarray:
        Apply random sepia tone effect
        if intensity is None:
            intensity = random.uniform(0.2, 0.8)
            
        sepia_matrix = np.array([
            [0.393 + 0.607 * (1 - intensity), 0.769 - 0.769 * (1 - intensity), 0.189 - 0.189 * (1 - intensity)],
            [0.349 - 0.349 * (1 - intensity), 0.686 + 0.314 * (1 - intensity), 0.168 - 0.168 * (1 - intensity)],
            [0.272 - 0.272 * (1 - intensity), 0.534 - 0.534 * (1 - intensity), 0.131 + 0.869 * (1 - intensity)]
        ])
        
        sepia_image = cv2.transform(image, sepia_matrix)
        return np.clip(sepia_image, 0, 255).astype(np.uint8) 
    
    
    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness and contrast"""
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur"""
        # Ensure kernel size is odd
        kernel_size = random.randrange(
            self.kernel_size_range[0], 
            self.kernel_size_range[1], 2
        )
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def apply_dilation(self, image: np.ndarray) -> np.ndarray:
         
        # Custom cross-shaped kernel for soft dilation
        kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        
        return cv2.dilate(image, kernel, iterations=1)


    def apply_erosion(self, image: np.ndarray) -> np.ndarray:
        
        kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        return cv2.erode(image, kernel, iterations=1) 
    
    def augment(self, image: Image.Image) -> dict:
        """Apply all color augmentations with random parameters"""
        # Convert PIL Image to numpy array
        img_np = np.array(image)
        
        # Store original
        augmented = {"original": img_np}
        
        # Apply different color transformations
        augmented["channel_shift"] = self.random_channel_shift(img_np)
        augmented["gamma"] = self.random_gamma_correction(img_np)
        augmented["channel_multiply"] = self.random_channel_multiplication(img_np)
        augmented["brightness_contrast"] = self.adjust_brightness_contrast(img_np)
        augmented["pytorch_color"] = np.array(self.pytorch_color_transforms(image))
        augmented["gaussian_blur"] = self.apply_gaussian_blur(img_np)
        
        
        return augmented

    def random_augmentation(self, image: Image.Image) -> np.ndarray:
        """Randomly select and apply one augmentation."""
        augmented = self.augment(image)
        # Randomly select one transformation type (skip "original")
        transform_type = random.choice(list(augmented.keys())[1:])
        #print(f"Selected transformation: {transform_type}")
        augmented_image = Image.fromarray(augmented[transform_type])
        return augmented_image

    @staticmethod
    def save_augmentations(augmented_images: dict, save_dir: str, base_filename: str):
        """Save all augmented images to specified directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        for aug_type, image in augmented_images.items():
            # Create filename
            filename = f"{base_filename}_{aug_type}.png"
            save_path = os.path.join(save_dir, filename)
            
            # Convert to BGR for OpenCV
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # Save image
            cv2.imwrite(save_path, image)
            print(f"Saved {aug_type} augmentation to: {save_path}")

    @staticmethod
    def visualize_augmentations(augmented_images: dict, save_path: str = None):
        """Visualize or save the augmented images"""
        n_images = len(augmented_images)
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for idx, (title, img) in enumerate(augmented_images.items(), 1):
            plt.subplot(n_rows, n_cols, idx)
            plt.title(title)
            plt.imshow(img)
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to: {save_path}")
        plt.show()

if __name__ == "__main__":
    # Initialize the augmentation class
    augmenter = ColorAugmentations()

    # Load an example image
    image_path = r""   # Path to an image in dataset
    image = Image.open(image_path).convert("RGB")

    # Get all augmentations
    augmented_images = augmenter.augment(image)

    # Visualize all augmentations
    augmenter.visualize_augmentations(augmented_images)

    # Save all augmentations
    save_dir = r""
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    augmenter.save_augmentations(augmented_images, save_dir, base_filename) 
    

    # Perform a random augmentation
    random_aug = augmenter.random_augmentation(image)
    plt.figure(figsize=(8, 8))
    plt.imshow(random_aug)
    plt.axis('off')
    plt.title("Random Single Augmentation")
    plt.show()
