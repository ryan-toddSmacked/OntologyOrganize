"""Image transformation functions for preprocessing."""

import numpy as np
from PIL import Image
from scipy import ndimage


def sobel_edge_detector(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection filter.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Edge-detected image as numpy array
    """
    # Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply convolution
    edge_x = ndimage.convolve(img_array.astype(np.float32), sobel_x)
    edge_y = ndimage.convolve(img_array.astype(np.float32), sobel_y)
    
    # Compute magnitude
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize to 0-255
    edge_magnitude = np.clip(edge_magnitude, 0, 255).astype(np.uint8)
    
    return edge_magnitude


def gaussian_blur(img_array: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur filter.
    
    Args:
        img_array: Grayscale image as numpy array
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image as numpy array
    """
    blurred = ndimage.gaussian_filter(img_array, sigma=sigma)
    return blurred.astype(np.uint8)


def laplacian_filter(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian filter for edge enhancement.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Filtered image as numpy array
    """
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Apply convolution
    filtered = ndimage.convolve(img_array.astype(np.float32), laplacian)
    
    # Normalize to 0-255
    filtered = np.clip(filtered + 128, 0, 255).astype(np.uint8)
    
    return filtered


def sharpen_filter(img_array: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Sharpened image as numpy array
    """
    # Sharpening kernel
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # Apply convolution
    sharpened = ndimage.convolve(img_array.astype(np.float32), sharpen)
    
    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def emboss_filter(img_array: np.ndarray) -> np.ndarray:
    """
    Apply emboss filter.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Embossed image as numpy array
    """
    # Emboss kernel
    emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    
    # Apply convolution
    embossed = ndimage.convolve(img_array.astype(np.float32), emboss)
    
    # Normalize to 0-255 range (adding 128 to center around middle gray)
    embossed = np.clip(embossed + 128, 0, 255).astype(np.uint8)
    
    return embossed


def invert_image(img_array: np.ndarray) -> np.ndarray:
    """
    Invert image colors.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Inverted image as numpy array
    """
    return (255 - img_array).astype(np.uint8)


def prewitt_edge_detector(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Prewitt edge detection filter.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Edge-detected image as numpy array
    """
    # Prewitt kernels for x and y directions
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply convolution
    edge_x = ndimage.convolve(img_array.astype(np.float32), prewitt_x)
    edge_y = ndimage.convolve(img_array.astype(np.float32), prewitt_y)
    
    # Compute magnitude
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize to 0-255
    edge_magnitude = np.clip(edge_magnitude, 0, 255).astype(np.uint8)
    
    return edge_magnitude


def median_filter(img_array: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply median filter for noise reduction.
    
    Args:
        img_array: Grayscale image as numpy array
        size: Size of the filter kernel
        
    Returns:
        Filtered image as numpy array
    """
    filtered = ndimage.median_filter(img_array, size=size)
    return filtered.astype(np.uint8)


def bilateral_filter(img_array: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving smoothing).
    Note: Using approximation with Gaussian filters.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Filtered image as numpy array
    """
    # Simplified bilateral using recursive Gaussian filtering
    from scipy.ndimage import gaussian_filter
    sigma_spatial = 3.0
    filtered = gaussian_filter(img_array.astype(np.float32), sigma=sigma_spatial)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def canny_edge_detector(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Edge-detected image as numpy array
    """
    # Simplified Canny using gradient magnitude and thresholding
    # Step 1: Gaussian smoothing
    smoothed = ndimage.gaussian_filter(img_array.astype(np.float32), sigma=1.4)
    
    # Step 2: Gradient calculation
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = ndimage.convolve(smoothed, sobel_x)
    gy = ndimage.convolve(smoothed, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Step 3: Double thresholding
    high_threshold = np.percentile(magnitude, 90)
    low_threshold = high_threshold * 0.4
    
    edges = np.zeros_like(magnitude)
    edges[magnitude >= high_threshold] = 255
    edges[(magnitude >= low_threshold) & (magnitude < high_threshold)] = 128
    
    return edges.astype(np.uint8)


def gradient_magnitude(img_array: np.ndarray) -> np.ndarray:
    """
    Calculate gradient magnitude.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Gradient magnitude as numpy array
    """
    # Calculate gradients in x and y
    gx = ndimage.sobel(img_array.astype(np.float32), axis=1)
    gy = ndimage.sobel(img_array.astype(np.float32), axis=0)
    
    # Compute magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Normalize to 0-255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def erosion(img_array: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological erosion.
    
    Args:
        img_array: Grayscale image as numpy array
        iterations: Number of erosion iterations
        
    Returns:
        Eroded image as numpy array
    """
    eroded = ndimage.grey_erosion(img_array, size=(3, 3))
    for _ in range(iterations - 1):
        eroded = ndimage.grey_erosion(eroded, size=(3, 3))
    return eroded.astype(np.uint8)


def dilation(img_array: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological dilation.
    
    Args:
        img_array: Grayscale image as numpy array
        iterations: Number of dilation iterations
        
    Returns:
        Dilated image as numpy array
    """
    dilated = ndimage.grey_dilation(img_array, size=(3, 3))
    for _ in range(iterations - 1):
        dilated = ndimage.grey_dilation(dilated, size=(3, 3))
    return dilated.astype(np.uint8)


def opening(img_array: np.ndarray) -> np.ndarray:
    """
    Apply morphological opening (erosion then dilation).
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Opened image as numpy array
    """
    opened = ndimage.grey_opening(img_array, size=(3, 3))
    return opened.astype(np.uint8)


def closing(img_array: np.ndarray) -> np.ndarray:
    """
    Apply morphological closing (dilation then erosion).
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Closed image as numpy array
    """
    closed = ndimage.grey_closing(img_array, size=(3, 3))
    return closed.astype(np.uint8)


def otsu_threshold(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        Binary thresholded image as numpy array
    """
    # Calculate histogram
    hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    
    # Normalize histogram
    hist_norm = hist / hist.sum()
    
    # Calculate cumulative sums
    cum_sum = np.cumsum(hist_norm)
    cum_mean = np.cumsum(hist_norm * np.arange(256))
    
    # Global mean
    global_mean = cum_mean[-1]
    
    # Calculate between-class variance
    variance = np.zeros(256)
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1.0 - w0
        
        if w0 == 0 or w1 == 0:
            continue
            
        m0 = cum_mean[t] / w0 if w0 > 0 else 0
        m1 = (global_mean - cum_mean[t]) / w1 if w1 > 0 else 0
        
        variance[t] = w0 * w1 * (m0 - m1) ** 2
    
    # Find optimal threshold
    threshold = np.argmax(variance)
    
    # Apply threshold
    binary = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    
    return binary


def adaptive_threshold(img_array: np.ndarray, block_size: int = 11) -> np.ndarray:
    """
    Apply adaptive thresholding.
    
    Args:
        img_array: Grayscale image as numpy array
        block_size: Size of local neighborhood
        
    Returns:
        Binary thresholded image as numpy array
    """
    # Calculate local mean using uniform filter
    local_mean = ndimage.uniform_filter(img_array.astype(np.float32), size=block_size)
    
    # Threshold: pixel value compared to local mean minus offset
    offset = 5
    binary = np.where(img_array > local_mean - offset, 255, 0).astype(np.uint8)
    
    return binary


def difference_of_gaussians(img_array: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0) -> np.ndarray:
    """
    Apply Difference of Gaussians (DoG) for blob detection.
    
    Args:
        img_array: Grayscale image as numpy array
        sigma1: Sigma for first Gaussian
        sigma2: Sigma for second Gaussian
        
    Returns:
        DoG filtered image as numpy array
    """
    gaussian1 = ndimage.gaussian_filter(img_array.astype(np.float32), sigma=sigma1)
    gaussian2 = ndimage.gaussian_filter(img_array.astype(np.float32), sigma=sigma2)
    
    dog = gaussian1 - gaussian2
    
    # Normalize to 0-255
    dog = dog - dog.min()
    dog = dog / dog.max() * 255 if dog.max() > 0 else dog
    
    return dog.astype(np.uint8)


def gabor_filter(img_array: np.ndarray, frequency: float = 0.15, theta: float = 0) -> np.ndarray:
    """
    Apply Gabor filter for texture/pattern detection.
    
    Args:
        img_array: Grayscale image as numpy array
        frequency: Frequency of the sinusoidal wave (0.1-0.3 typical)
        theta: Orientation in radians
        
    Returns:
        Gabor filtered image as numpy array
    """
    from scipy.ndimage import convolve
    
    # Create Gabor kernel
    kernel_size = 21
    sigma = 4.0
    gamma = 0.5  # Spatial aspect ratio
    
    # Create coordinate grid
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    X, Y = np.meshgrid(x, y)
    
    # Rotate coordinates
    x_theta = X * np.cos(theta) + Y * np.sin(theta)
    y_theta = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Gabor function (with gamma for elliptical envelope)
    gaussian_envelope = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta)
    gabor = gaussian_envelope * sinusoid
    
    # Apply convolution
    filtered = convolve(img_array.astype(np.float32), gabor)
    
    # Normalize using mean and std for better contrast
    filtered = np.abs(filtered)  # Take magnitude
    mean = filtered.mean()
    std = filtered.std()
    
    # Normalize to approximately 0-255 using z-score normalization
    if std > 0:
        filtered = (filtered - mean) / std
        filtered = np.clip(filtered * 50 + 128, 0, 255)  # Scale and center
    else:
        filtered = np.full_like(filtered, 128)
    
    return filtered.astype(np.uint8)


def high_pass_filter(img_array: np.ndarray) -> np.ndarray:
    """
    Apply high-pass filter to emphasize edges and details.
    
    Args:
        img_array: Grayscale image as numpy array
        
    Returns:
        High-pass filtered image as numpy array
    """
    # Low-pass filter (blur)
    low_pass = ndimage.gaussian_filter(img_array.astype(np.float32), sigma=2.0)
    
    # High-pass = Original - Low-pass
    high_pass = img_array.astype(np.float32) - low_pass
    
    # Shift to positive range and normalize
    high_pass = high_pass + 128
    high_pass = np.clip(high_pass, 0, 255)
    
    return high_pass.astype(np.uint8)


def apply_transform(img_array: np.ndarray, transform_name: str) -> np.ndarray:
    """
    Apply a named transformation to an image.
    
    Args:
        img_array: Grayscale image as numpy array
        transform_name: Name of the transformation to apply
        
    Returns:
        Transformed image as numpy array
    """
    transform_map = {
        'sobel': sobel_edge_detector,
        'prewitt': prewitt_edge_detector,
        'canny': canny_edge_detector,
        'gradient': gradient_magnitude,
        'gaussian_blur': gaussian_blur,
        'median': median_filter,
        'bilateral': bilateral_filter,
        'laplacian': laplacian_filter,
        'sharpen': sharpen_filter,
        'emboss': emboss_filter,
        'invert': invert_image,
        'erosion': erosion,
        'dilation': dilation,
        'opening': opening,
        'closing': closing,
        'otsu': otsu_threshold,
        'adaptive': adaptive_threshold,
        'dog': difference_of_gaussians,
        'gabor': gabor_filter,
        'highpass': high_pass_filter,
        'none': lambda x: x,  # No transformation
    }
    
    if transform_name in transform_map:
        return transform_map[transform_name](img_array)
    else:
        print(f"Warning: Unknown transformation '{transform_name}', returning original")
        return img_array
