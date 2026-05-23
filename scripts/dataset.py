import json
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
import torchvision.transforms.functional as TF
import numpy as np
import random
import copy
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from collections import Counter

def create_dataloaders(data_dir, batch_size=8, train_split=0.8, patch_size=512, 
                       num_workers=4, center_bias=0.0, three_channel=False, 
                       drop_classes=[-5,-4,-3,-2], label_key='category', 
                       evenly_sample=True
    ):
    """
    Create train and test dataloaders.
    
    Args:
        image_metadata_path: Directory containing NetCDF files
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training (default 0.8)
        patch_size: Size of image patches
        num_workers: Number of workers for data loading
        center_bias: Controls crop location bias toward center (0.0=uniform, 1.0=center only)
        three_channel (bool): Convert grayscale imagery to 3 channel RGB
        drop_classes (list): Drop IBTrACS observations with cerain class IDs
        label_key (str): The column from image_metadata_path to use as the label in a batch
        evenly_sample (bool): Sample across classes evenly rather than from original distribution

    Returns:
        train_loader, val_loader
    """

    # Create dataset
    full_dataset = GOESDataset(
        data_dir, 
        patch_size=patch_size, 
        augment=True, 
        center_bias=center_bias,
        three_channel=three_channel,
        drop_classes=drop_classes, 
        label_key=label_key
    )
    
    # Load full metadata to get indices first
    n = len(full_dataset)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    split = int(train_split * n)
    train_indices, val_indices = indices[:split], indices[split:]

    # Two separate instances — different augment settings, same underlying files
    train_dataset = GOESDataset(
        data_dir, 
        patch_size=patch_size, 
        augment=True, 
        center_bias=center_bias,
        three_channel=three_channel,
        drop_classes=drop_classes, 
        label_key=label_key
    )
    val_dataset   = GOESDataset(
        data_dir, 
        patch_size=patch_size, 
        augment=False, 
        center_bias=center_bias,
        three_channel=three_channel,
        drop_classes=drop_classes, 
        label_key=label_key
    )
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset   = Subset(val_dataset,   val_indices)
    
    if evenly_sample:
        train_sampler = create_balanced_sampler(train_dataset,label_key=label_key)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    # Create test dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def save_batch_mosaic(batch,filepath='./training_sample.png'):
    
    """
    Save a batch to a mosiac

    Args:
        batch
    """

    batch_size = len(batch['label'])
    patch_size = batch['patch'][0].shape[-1]
    ndims = len(batch['patch'][0].shape) 

    # Calculate mosaic dimensions
    padding = 3
    mosaic_width = 4
    mosaic_height = math.ceil(batch_size / mosaic_width)
    mosaic_img_width = mosaic_width * patch_size + (mosaic_width + 1) * padding
    mosaic_img_height = mosaic_height * patch_size + (mosaic_height + 1) * padding

    # Create empty mosaic array (white background)
    if ndims == 2:
        mosaic = np.ones((mosaic_img_height, mosaic_img_width), dtype=np.uint8) * 0
    else:
        mosaic = np.ones((3, mosaic_img_height, mosaic_img_width), dtype=np.uint8) * 0

    # Randomly crop the tile
    for i in range(mosaic_width):
        for j in range(mosaic_height):

            # Iterate through the dataset
            batch_ind = i*mosaic_height + j 
            patch = batch['patch'][batch_ind]
            category = batch['label'][batch_ind]

            # Calculate position in mosaic
            y_start = padding + j * (patch_size + padding)
            x_start = padding + i * (patch_size + padding)
            
            # Place tile in mosaic
            mosaic[..., y_start:y_start + patch_size, x_start:x_start + patch_size] = patch*255

    # Convert to PIL Image
    if ndims==3:
        mosaic_img = Image.fromarray(mosaic.transpose(1, 2, 0))
    else:
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
            category = batch['label'][batch_ind]
            
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

    mosaic_img.save(filepath)

    return mosaic_img

def create_balanced_sampler(dataset, label_key='category', negative_class_ratio=0.5):
    """
    Create a WeightedRandomSampler to balance classes during training.
    Automatically extracts label_map from the dataset.
    
    Args:
        dataset: PyTorch Dataset or Subset
        negative_class_ratio: Proportion of samples that should be from the negative class ("-9999")
                              Default 0.5 means 50% negative, 50% positive (split among other classes)
    
    Returns:
        WeightedRandomSampler
    """
    # Get the base dataset and extract label_map
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))
    
    # Get all unique categories from the dataset
    all_categories = set()
    for idx in range(len(base_dataset)): # type: ignore
        category_str = base_dataset.image_metadata['images'][idx][label_key] # type: ignore
        all_categories.add(category_str)
    
    # Create label map (sorted for consistency)
    sorted_categories = sorted(all_categories)
    label_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
    
    # Find the negative class index
    negative_class_idx = label_map.get(-9999, None)
    if negative_class_idx is None:
        negative_class_ratio = None
    
    # Get all labels from the dataset
    labels = []
    for idx in indices:
        category_str = base_dataset.image_metadata['images'][idx][label_key] # type: ignore
        labels.append(label_map[category_str])
    
    # Count samples per class
    class_counts = Counter(labels)

    # Calculate weight for each class
    num_classes = len(class_counts)
    class_weights = {}
    
    if negative_class_ratio is not None:
        # Custom weighting: negative_class_ratio for "-9999", rest split among other classes
        num_positive_classes = num_classes - 1  # All classes except "-9999"
        positive_class_ratio = (1.0 - negative_class_ratio) / num_positive_classes
        
        for class_idx in range(num_classes):
            if class_idx not in class_counts:
                class_weights[class_idx] = 0.0
                continue
                
            if class_idx == negative_class_idx:
                # Weight for negative class
                # target_ratio / actual_ratio gives us the weight
                actual_ratio = class_counts[class_idx] / len(labels)
                class_weights[class_idx] = negative_class_ratio / actual_ratio
            else:
                # Weight for positive classes
                actual_ratio = class_counts[class_idx] / len(labels)
                class_weights[class_idx] = positive_class_ratio / actual_ratio
    else:
        # Standard balanced sampling (all classes equal)
        for class_idx in range(num_classes):
            if class_idx in class_counts:
                class_weights[class_idx] = len(labels) / (num_classes * class_counts[class_idx])
            else:
                class_weights[class_idx] = 0.0
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

class GOESDataset(Dataset):
    """PyTorch Dataset for GOES satellite NetCDF imagery."""
    
    def __init__(self, image_metadata_path, patch_size=512, augment=True, center_bias=0.5, three_channel=False, drop_classes=[-5,-4,-3,-2], label_key='category'):
        """
        Args:
            image_metadata_path (str): JSON image metadata file pointing to image paths 
            patch_size (int): Size of random crop (default 512x512)
            augment (True): Whether to apply augmentations
            center_bias (float): Controls crop location bias toward center (0.0=uniform, 1.0=center only)
            three_channel (bool): Convert grayscale imagery to 3 channel RGB
            drop_classes (list): Drop IBTrACS observations with cerain class IDs
            label_key (str): The column from image_metadata_path to use as the label in a batch

        Tropical storm category IDs from IBTrACS:
            -5 = Unknown [XX]
            -4 = Post-tropical [EX, ET, PT]
            -3 = Miscellaneous disturbances [WV, LO, DB, DS, IN, MD]
            -2 = Subtropical [SS, SD]
            Tropical systems classified based on wind speeds [TD, TS, HU, TY,, TC, ST, HR]
            -1 = Tropical depression (W<34)
            0 = Tropical storm [34<W<64]
            1 = Category 1 [64<=W<83]
            2 = Category 2 [83<=W<96]
            3 = Category 3 [96<=W<113]
            4 = Category 4 [113<=W<137]
            26 USA_SSHS (same)
            5 = Category 5 [W >= 137]

        """
        self.image_metadata_path = image_metadata_path
        self.patch_size = patch_size
        self.augment = augment
        self.center_bias = center_bias
        self.three_channel = three_channel
        self.drop_classes = drop_classes
        self.label_key = label_key

        # Load the image metadata
        self.image_metadata = self._load_image_metadata()

        # Drop classes from positive samples if specified
        if self.drop_classes:
            self._drop_classes()

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

    def _drop_classes(self):
        if self.image_metadata is None:
            raise ValueError('image_metadata has not been loaded')
        if self.drop_classes is not None:
            tmp = copy.deepcopy(self.image_metadata)
            tmp['images'] = []
            for img_info in self.image_metadata['images']:
                if (img_info['sshs'] not in self.drop_classes):
                    tmp['images'].append(img_info)
            self.image_metadata = tmp

    def _load_netcdf(self,filepath,invert=False):
        """Load data from NetCDF file."""
        # Read and normalize data
        data = xr.open_dataset(filepath)
        rad_data = data.Rad.values
        rad_data = np.where(np.isnan(rad_data),0.0,rad_data)
        data.close()

        # Normalize to 0-255
        rad_normalized = (rad_data - np.nanmin(rad_data)) / (np.nanmax(rad_data) - np.nanmin(rad_data))
        rad_normalized = np.where(rad_normalized==np.nan,0.0,rad_normalized)
        rad_uint8 = (rad_normalized * 255).astype(np.uint8)

        if invert:
            rad_uint8 = 255 - rad_uint8

        return rad_uint8

    def _random_crop(self, arr):
        """Extract random patch from image, with optional center bias."""
        # h, w = arr.shape[-2:]
        if arr.ndim == 3:
            _, h, w = arr.shape
        else:
            h, w = arr.shape


        if h < self.patch_size or w < self.patch_size:
            # Pad if image is smaller than patch size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
            # h, w = arr.shape[-2:]
            if arr.ndim == 3:
                _, h, w = arr.shape
            else:
                h, w = arr.shape
        
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
            arr = torch.clamp(arr, 0, 255)
        
        return arr
    
    def _normalize(self, t):
        mx = t.max()
        if mx <= 0:
            return torch.zeros_like(t)  # explicitly handle bad patches
        return torch.clamp(t / mx, 0, 1)

    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.image_metadata is None:
            raise ValueError('image_metadata has not been loaded')
        return len(self.image_metadata['images'])

    def __getitem__(self, idx):
        """Load and process a single sample."""
        if self.image_metadata is None:
            raise ValueError('image_metadata has not been loaded')
        image_info = self.image_metadata['images'][idx]
        filepath = image_info['file_name']
        label = image_info[self.label_key]
        
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
        # # patch = (patch - patch.mean()) / (patch.std() + 1e-8)
        # patch = patch / patch.max()
        # patch = torch.clamp(patch, 0, 1)
        patch = self._normalize(patch)

        # Repeat to three channel RGB
        if self.three_channel:
            if patch.shape[0] == 1:
                patch = patch.repeat(3, 1, 1)
        
        return {'patch':patch,'label':str(label)}
    

if __name__=='__main__':

    # Set arguments for the data loaders
    # The path to the JSON file for downloaded image metadata
    image_metadata_path = '/Users/dylanwhite/Projects/tropical-cv/data/training/image_data.json'
    # The patch size to subset from the downloaded netCDF files
    patch_size = 512
    # The batch size
    batch_size = 16
    # How biased to be towards the center of a tile for positive tiles
    center_bias = 0.6
    # Convert to RGB or leave as greyscale
    three_channel=False
    # Drop Saffir-Simpson classes from IBTrACS observations
    drop_classes=[-5,-4,-3,-2]
    # The label for classes drawn
    label_key='sshs'

    # Create a train/test split
    train_loader, test_loader = create_dataloaders(
        image_metadata_path, 
        batch_size=batch_size, 
        train_split=0.8, 
        patch_size=patch_size, 
        num_workers=0, 
        center_bias=center_bias,
        label_key=label_key,
        three_channel=three_channel,
        drop_classes=drop_classes,
        evenly_sample=True
    )

    # Get iterator
    data_iter = iter(train_loader)

    # Get a batch from the loader
    batch = next(data_iter)

    mosaic_img = save_batch_mosaic(batch)

    plt.imshow(mosaic_img,cmap="Greys")
    plt.axis('off')
    plt.show()