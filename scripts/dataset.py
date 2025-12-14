import json
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import numpy as np
import random

def create_dataloaders(data_dir, batch_size=8, train_split=0.8, patch_size=512, 
                       num_workers=4, center_bias=0.0):
    """
    Create train and test dataloaders.
    
    Args:
        image_metadata_path: Directory containing NetCDF files
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training (default 0.8)
        patch_size: Size of image patches
        num_workers: Number of workers for data loading
        center_bias: Controls crop location bias toward center (0.0=uniform, 1.0=center only)
    
    Returns:
        train_loader, test_loader
    """

    # Create dataset
    dataset = GOESDataset(
        data_dir, 
        patch_size=patch_size, 
        augment=True, 
        center_bias=center_bias
    )
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    return train_loader, test_loader

class GOESDataset(Dataset):
    """PyTorch Dataset for GOES satellite NetCDF imagery."""
    
    def __init__(self, image_metadata_path, patch_size=512, augment=True, center_bias=0.5, three_channel=False):
        """
        Args:
            image_metadata_path (str): JSON image metadata file pointing to image paths 
            patch_size (int): Size of random crop (default 512x512)
            augment (True): Whether to apply augmentations
            center_bias (float): Controls crop location bias toward center (0.0=uniform, 1.0=center only)
            three_channel (bool): Convert grayscale imagery to 3 channel RGB
        """
        self.image_metadata_path = image_metadata_path
        self.patch_size = patch_size
        self.augment = augment
        self.center_bias = center_bias
        self.three_channel = three_channel

        # Load the image metadata
        self.image_metadata = self._load_image_metadata()

    def _load_image_metadata(self):
        """Load the JSON metadata file for processed GOES imagery tiles."""
        with open(self.image_metadata_path,'r') as f:
            image_metadata = json.load(f)
        
        # Filter out images with invalid dimensions
        valid_images = []
        for img_info in image_metadata['images']:
            filepath = img_info['file_name']
            try:
                data = xr.open_dataset(filepath)
                shape = data.Rad.shape
                data.close()
                
                if shape[0] > 0 and shape[1] > 0:
                    valid_images.append(img_info)
                else:
                    print(f"Skipping {filepath} with invalid shape {shape}")
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}")
        
        image_metadata['images'] = valid_images
        assert len(image_metadata['images']) > 0, "There are no valid images in the image metadata file."
        return image_metadata

    def _load_netcdf(self,filepath,invert=False):
        """Load data from NetCDF file."""
        # Read and normalize data
        data = xr.open_dataset(filepath)
        rad_data = data.Rad.values
        rad_data = np.where(np.isnan(rad_data),0.0,rad_data)
        data.close()

        # Normalize to 0-255
        rad_normalized = (rad_data - rad_data.min()) / (rad_data.max() - rad_data.min())
        rad_uint8 = (rad_normalized * 255).astype(np.uint8)

        if invert:
            rad_uint8 = 255 - rad_uint8

        return rad_uint8

    def _random_crop(self, arr):
        """Extract random patch from image, with optional center bias."""
        h, w = arr.shape[-2:]
        
        if h < self.patch_size or w < self.patch_size:
            # Pad if image is smaller than patch size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = arr.shape[-2:]
        
        # Calculate center of image
        center_y = h // 2
        center_x = w // 2
        
        # Calculate maximum offset from center
        max_offset_y = center_y - self.patch_size // 2
        max_offset_x = center_x - self.patch_size // 2
        
        if self.center_bias == 0.0:
            # Uniform random crop (original behavior)
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
        else:
            # Biased toward center
            # Reduce the sampling range based on center_bias
            # At center_bias=1.0, the range shrinks to 0 (center only)
            range_y = int(max_offset_y * (1.0 - self.center_bias))
            range_x = int(max_offset_x * (1.0 - self.center_bias))
            
            # Sample offset from center within reduced range
            offset_y = random.randint(-range_y, range_y) if range_y > 0 else 0
            offset_x = random.randint(-range_x, range_x) if range_x > 0 else 0
            
            # Calculate top-left corner
            top = center_y - self.patch_size // 2 + offset_y
            left = center_x - self.patch_size // 2 + offset_x
            
            # Clamp to valid range (safety check)
            top = max(0, min(top, h - self.patch_size))
            left = max(0, min(left, w - self.patch_size))
        
        return arr[top:top + self.patch_size, left:left + self.patch_size]

    def _augment(self, arr):
        """Apply random augmentations."""
        # Convert to torch tensor
        arr = torch.from_numpy(arr).float()
        
        # Add channel dimension if needed
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        
        # Random horizontal flip
        if random.random() > 0.5:
            arr = TF.hflip(arr)
        
        # Random vertical flip
        if random.random() > 0.5:
            arr = TF.vflip(arr)
        
        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            arr = TF.rotate(arr, angle)
        
        # Random brightness/contrast (rescaling)
        if random.random() > 0.5:
            # Apply random scaling factor
            scale = random.uniform(0.6, 1.4)
            arr = arr * scale
            #arr = np.clip(arr,0,255)
            arr = torch.clamp(arr, 0, 255)
        
        return arr

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_metadata['images'])

    def __getitem__(self, idx):
        """Load and process a single sample."""
        image_info = self.image_metadata['images'][idx]
        filepath = image_info['file_name']
        category = image_info['category']
        
        # Load NetCDF data
        img = self._load_netcdf(filepath)
        
        # Extract random patch
        patch = self._random_crop(img)
        
        # Apply augmentations
        if self.augment:
            patch = self._augment(patch)
        else:
            patch = torch.from_numpy(patch).float()
            if patch.ndim == 2:
                patch = patch.unsqueeze(0)
        
        # Normalize
        # patch = (patch - patch.mean()) / (patch.std() + 1e-8)
        patch = patch / patch.max()
        patch = torch.clamp(patch, 0, 1)

        # Repeat to three channel RGB
        if patch.shape[0] == 1:
            patch = patch.repeat(3, 1, 1)
        
        return {'patch':patch,'category':category,'metadata':image_info}
    
if __name__=='__main__':

    # Imports
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import math

    # Set arguments for the data loaders
    # The path to the JSON file for downloaded image metadata
    image_metadata_path = '/Users/dylanwhite/Documents/Projects/tropical-cv/data/training/image_data.json'
    # The patch size to subset from the downloaded netCDF files
    patch_size = 512
    # The batch size
    batch_size = 16
    # How biased to be towards the center of a tile for positive tiles
    center_bias = 0.6

    # Create a train/test split
    train_loader, test_loader = create_dataloaders(
        image_metadata_path, 
        batch_size=batch_size, 
        train_split=0.8, 
        patch_size=patch_size, 
        num_workers=0, 
        center_bias=center_bias
    )

    # Get iterator
    data_iter = iter(train_loader)

    # Get a batch from the loader
    batch = next(data_iter)

    # Calculate mosaic dimensions
    padding = 3
    mosaic_width = 4
    mosaic_height = math.ceil(batch_size / mosaic_width)
    mosaic_img_width = mosaic_width * patch_size + (mosaic_width + 1) * padding
    mosaic_img_height = mosaic_height * patch_size + (mosaic_height + 1) * padding

    # Create empty mosaic array (white background)
    mosaic = np.ones((mosaic_img_height, mosaic_img_width), dtype=np.uint8) * 0

    # Randomly crop the tile
    for i in range(mosaic_width):
        for j in range(mosaic_height):

            # Iterate through the dataset
            batch_ind = i*mosaic_height + j 
            patch = batch['patch'][batch_ind]
            category = batch['category'][batch_ind]

            # Calculate position in mosaic
            y_start = padding + j * (patch_size + padding)
            x_start = padding + i * (patch_size + padding)
            
            # Place tile in mosaic
            mosaic[y_start:y_start + patch_size, x_start:x_start + patch_size] = patch*255

    # Convert to PIL Image (move this outside the loop)
    mosaic_img = Image.fromarray(mosaic)

    # Add text labels
    draw = ImageDraw.Draw(mosaic_img)

    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default(size=30)

    # Add text to each patch
    for i in range(mosaic_width):
        for j in range(mosaic_height):
            batch_ind = i*mosaic_height + j
            category = batch['category'][batch_ind]
            
            # Calculate position for text (top-left corner of each patch)
            x_start = padding + i * (patch_size + padding) + 5
            y_start = padding + j * (patch_size + padding) + 5
            
            # Draw text with outline for better visibility
            # Draw black outline
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    draw.text((x_start + offset_x, y_start + offset_y), category, fill=0, font=font)
            # Draw white text on top
            draw.text((x_start, y_start), category, fill=255, font=font)

    plt.imshow(mosaic_img,cmap="Greys")
    plt.axis('off')
    plt.show()