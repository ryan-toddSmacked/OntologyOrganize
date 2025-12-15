"""Utility functions for image processing."""

from pathlib import Path
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import io

from src.utils.image_transforms import apply_transform


def create_thumbnail(image_path: Path, size: tuple[int, int] = (100, 100), colormap: str = "gray", transform: str = "none", cached_image: Image.Image = None) -> QPixmap | None:
    """
    Create a thumbnail QPixmap from an image file or cached image.
    
    Args:
        image_path: Path to image file
        size: Thumbnail size (width, height)
        colormap: Colormap to apply (default: "gray")
        transform: Image transformation to apply (default: "none")
        cached_image: Optional pre-loaded PIL Image to use instead of loading from disk
        
    Returns:
        QPixmap thumbnail or None if error
    """
    try:
        # Use cached image if available, otherwise load from disk
        if cached_image is not None:
            img = cached_image.copy()  # Make a copy to avoid modifying cached version
        else:
            img = Image.open(image_path)
        
        # Always convert to grayscale if image has multiple channels
        # Handles RGB, RGBA, CMYK, and other multi-channel formats
        if img.mode != 'L':
            print(f"Converting {image_path.name} from {img.mode} to grayscale")
            img = img.convert('L')
        
        # Apply transformation if specified
        if transform != "none":
            img_array = np.array(img)
            img_array = apply_transform(img_array, transform)
            img = Image.fromarray(img_array, 'L')
        
        # Apply colormap if not gray
        if colormap != "gray":
            img_array = np.array(img)
            img = apply_colormap(img_array, colormap)
        
        # Create thumbnail
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert PIL image to QPixmap
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(img_byte_arr.read())
        
        return pixmap
        
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None


def apply_colormap(gray_image: np.ndarray, colormap_name: str) -> Image.Image:
    """
    Apply a colormap to a grayscale image.
    
    Args:
        gray_image: Grayscale image as numpy array
        colormap_name: Name of colormap to apply
        
    Returns:
        PIL Image with colormap applied
    """
    try:
        # Try cmcrameri colormaps first
        import cmcrameri.cm as cmc
        import matplotlib.pyplot as plt
        
        # Get the colormap
        cmap_dict = {
            'batlow': cmc.batlow,
            'berlin': cmc.berlin,
            'broc': cmc.broc,
            'cork': cmc.cork,
            'hawaii': cmc.hawaii,
            'imola': cmc.imola,
            'lajolla': cmc.lajolla,
            'lapaz': cmc.lapaz,
            'nuuk': cmc.nuuk,
            'oslo': cmc.oslo,
            'roma': cmc.roma,
            'tokyo': cmc.tokyo,
            'turku': cmc.turku,
            'vik': cmc.vik,
        }
        
        if colormap_name in cmap_dict:
            cmap = cmap_dict[colormap_name]
        else:
            # Fall back to matplotlib colormaps
            cmap = plt.get_cmap(colormap_name)
        
        # Normalize image to 0-1 range
        normalized = gray_image.astype(np.float32) / 255.0
        
        # Apply colormap
        colored = cmap(normalized)
        
        # Convert to 0-255 RGB
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return Image.fromarray(rgb, 'RGB')
        
    except Exception as e:
        print(f"Error applying colormap {colormap_name}: {e}")
        # Return original as grayscale
        return Image.fromarray(gray_image, 'L')


def _load_image(img_path: Path, image_cache: dict = None) -> Image.Image:
    """
    Load an image from cache or disk.
    
    Args:
        img_path: Path to image file
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        PIL Image in grayscale
    """
    # Try to get from cache first
    if image_cache is not None:
        cached_img = image_cache.get(str(img_path))
        if cached_img is not None:
            return cached_img.copy()
    
    # Load from disk
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')
    return img


def compute_image_correlation(img1_path: Path, img2_path: Path, method: str = 'ncc', transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute similarity score between two images.
    
    Args:
        img1_path: Path to first image (base image)
        img2_path: Path to second image (comparison image)
        method: Correlation method ('ncc' for normalized cross-correlation, 'mse' for mean squared error)
        transform: Image transformation to apply before correlation (default: 'none')
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity score (higher is more similar for ncc, lower is more similar for mse)
    """
    try:
        # Load both images (from cache if available)
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        # Resize img2 to match img1 dimensions
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # Apply transformation if specified
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        if method == 'ncc':
            # Normalized Cross-Correlation
            # Normalize both arrays
            arr1_norm = (arr1 - arr1.mean()) / (arr1.std() + 1e-10)
            arr2_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-10)
            
            # Compute correlation
            correlation = np.mean(arr1_norm * arr2_norm)
            return float(correlation)
            
        elif method == 'mse':
            # Mean Squared Error (lower is better, so we return negative for consistency)
            mse = np.mean((arr1 - arr2) ** 2)
            return float(-mse)
        
        else:
            # Default to NCC
            arr1_norm = (arr1 - arr1.mean()) / (arr1.std() + 1e-10)
            arr2_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-10)
            correlation = np.mean(arr1_norm * arr2_norm)
            return float(correlation)
            
    except Exception as e:
        print(f"Error computing correlation between {img1_path} and {img2_path}: {e}")
        return 0.0


def compute_ssim(img1_path: Path, img2_path: Path) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        SSIM score (between -1 and 1, higher is more similar)
    """
    try:
        from scipy.ndimage import gaussian_filter
        
        # Load both images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Convert to grayscale
        if img1.mode != 'L':
            img1 = img1.convert('L')
        if img2.mode != 'L':
            img2 = img2.convert('L')
        
        # Resize img2 to match img1 dimensions
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Compute means
        mu1 = gaussian_filter(arr1, sigma=1.5)
        mu2 = gaussian_filter(arr2, sigma=1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = gaussian_filter(arr1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = gaussian_filter(arr2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = gaussian_filter(arr1 * arr2, sigma=1.5) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
        
    except Exception as e:
        print(f"Error computing SSIM between {img1_path} and {img2_path}: {e}")
        return 0.0


def compute_histogram_correlation(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute histogram correlation between two images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply before correlation
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Histogram correlation score (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute histograms
        hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Normalize
        hist1 = hist1.astype(np.float32) / hist1.sum()
        hist2 = hist2.astype(np.float32) / hist2.sum()
        
        # Compute correlation
        corr = np.corrcoef(hist1, hist2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error computing histogram correlation: {e}")
        return 0.0


def compute_chi_square_distance(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute chi-square distance between histograms (lower is more similar).
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative chi-square distance (higher is more similar for consistency)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Add small epsilon to avoid division by zero
        hist1 = hist1.astype(np.float32) + 1e-10
        hist2 = hist2.astype(np.float32) + 1e-10
        
        # Chi-square distance
        chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
        
        # Return negative for consistency (higher = more similar)
        return float(-chi_square)
        
    except Exception as e:
        print(f"Error computing chi-square distance: {e}")
        return 0.0


def compute_bhattacharyya_distance(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute Bhattacharyya distance between histograms.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative Bhattacharyya distance (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))
        
        # Normalize
        hist1 = hist1.astype(np.float32) / hist1.sum()
        hist2 = hist2.astype(np.float32) / hist2.sum()
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))
        
        # Bhattacharyya distance
        distance = -np.log(bc + 1e-10)
        
        # Return negative for consistency
        return float(-distance)
        
    except Exception as e:
        print(f"Error computing Bhattacharyya distance: {e}")
        return 0.0


def compute_emd(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein distance) between histograms.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative EMD (higher is more similar)
    """
    try:
        from scipy.stats import wasserstein_distance
        
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute EMD on flattened arrays
        emd = wasserstein_distance(arr1.flatten(), arr2.flatten())
        
        # Return negative for consistency
        return float(-emd)
        
    except Exception as e:
        print(f"Error computing EMD: {e}")
        return 0.0


def compute_mae(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Negative MAE (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        mae = np.mean(np.abs(arr1 - arr2))
        return float(-mae)
        
    except Exception as e:
        print(f"Error computing MAE: {e}")
        return 0.0


def compute_cosine_similarity(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute cosine similarity between images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Cosine similarity (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        if transform != 'none':
            arr1 = apply_transform(arr1.astype(np.uint8), transform).astype(np.float32)
            arr2 = apply_transform(arr2.astype(np.uint8), transform).astype(np.float32)
        
        # Flatten arrays
        vec1 = arr1.flatten()
        vec2 = arr2.flatten()
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        similarity = dot_product / (norm1 * norm2 + 1e-10)
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return 0.0


def compute_mutual_information(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute mutual information between images.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Mutual information (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(arr1.flatten(), arr2.flatten(), bins=256)
        
        # Compute marginal distributions
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        # Compute mutual information
        px_py = px[:, None] * py[None, :]
        
        # Avoid log(0)
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-10)))
        
        return float(mi)
        
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        return 0.0


def compute_hog_similarity(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute similarity based on Histogram of Oriented Gradients.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        HOG similarity (higher is more similar)
    """
    try:
        from scipy.ndimage import gaussian_filter
        
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
        
        # Compute gradients
        gx1 = np.gradient(arr1.astype(float), axis=1)
        gy1 = np.gradient(arr1.astype(float), axis=0)
        gx2 = np.gradient(arr2.astype(float), axis=1)
        gy2 = np.gradient(arr2.astype(float), axis=0)
        
        # Compute gradient magnitudes and orientations
        mag1 = np.sqrt(gx1**2 + gy1**2)
        ori1 = np.arctan2(gy1, gx1)
        mag2 = np.sqrt(gx2**2 + gy2**2)
        ori2 = np.arctan2(gy2, gx2)
        
        # Create orientation histograms (9 bins)
        bins = 9
        hist1, _ = np.histogram(ori1.flatten(), bins=bins, range=(-np.pi, np.pi), weights=mag1.flatten())
        hist2, _ = np.histogram(ori2.flatten(), bins=bins, range=(-np.pi, np.pi), weights=mag2.flatten())
        
        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # Compute correlation
        corr = np.corrcoef(hist1, hist2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error computing HOG similarity: {e}")
        return 0.0


def compute_perceptual_hash(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute perceptual hash similarity.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity based on hash distance (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
            img1 = Image.fromarray(arr1)
            img2 = Image.fromarray(arr2)
        
        # Resize to 8x8
        img1_small = img1.resize((8, 8), Image.Resampling.LANCZOS)
        img2_small = img2.resize((8, 8), Image.Resampling.LANCZOS)
        
        arr1_small = np.array(img1_small, dtype=np.float64)
        arr2_small = np.array(img2_small, dtype=np.float64)
        
        # Compute DCT (simplified using mean as threshold)
        mean1 = arr1_small.mean()
        mean2 = arr2_small.mean()
        
        hash1 = arr1_small > mean1
        hash2 = arr2_small > mean2
        
        # Hamming distance
        hamming = np.sum(hash1 != hash2)
        
        # Convert to similarity (0-64 bits, higher similarity = lower hamming)
        similarity = 1.0 - (hamming / 64.0)
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing perceptual hash: {e}")
        return 0.0


def compute_difference_hash(img1_path: Path, img2_path: Path, transform: str = 'none', image_cache: dict = None) -> float:
    """
    Compute difference hash similarity.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        transform: Image transformation to apply
        image_cache: Optional dict of cached images {path: PIL.Image}
        
    Returns:
        Similarity based on dHash (higher is more similar)
    """
    try:
        img1 = _load_image(img1_path, image_cache)
        img2 = _load_image(img2_path, image_cache)
        
        arr1 = np.array(img1, dtype=np.uint8)
        arr2 = np.array(img2, dtype=np.uint8)
        
        if transform != 'none':
            arr1 = apply_transform(arr1, transform)
            arr2 = apply_transform(arr2, transform)
            img1 = Image.fromarray(arr1)
            img2 = Image.fromarray(arr2)
        
        # Resize to 9x8
        img1_small = img1.resize((9, 8), Image.Resampling.LANCZOS)
        img2_small = img2.resize((9, 8), Image.Resampling.LANCZOS)
        
        arr1_small = np.array(img1_small, dtype=np.float64)
        arr2_small = np.array(img2_small, dtype=np.float64)
        
        # Compute horizontal gradient
        hash1 = arr1_small[:, 1:] > arr1_small[:, :-1]
        hash2 = arr2_small[:, 1:] > arr2_small[:, :-1]
        
        # Hamming distance
        hamming = np.sum(hash1 != hash2)
        
        # Convert to similarity
        similarity = 1.0 - (hamming / 64.0)
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing difference hash: {e}")
        return 0.0

        return Image.fromarray(gray_image, 'L')
