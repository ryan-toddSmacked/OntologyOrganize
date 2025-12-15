"""Download 500 MNIST digit images for testing."""

import gzip
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image


def download_file(url: str, destination: Path):
    """Download a file from URL to destination."""
    print(f"  Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, destination)


def parse_mnist_images(filepath: Path) -> np.ndarray:
    """Parse MNIST image file format."""
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        
    return data


def parse_mnist_labels(filepath: Path) -> np.ndarray:
    """Parse MNIST label file format."""
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and count
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def download_mnist_images(output_dir: str, num_images: int = 500):
    """
    Download and save MNIST digit images.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to download (default: 500)
    """
    print(f"Downloading {num_images} MNIST images...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for downloaded files
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # MNIST dataset URLs (using reliable mirror)
        base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
        }
        
        # Download files
        print("Downloading MNIST dataset...")
        train_images_path = temp_dir / files["train_images"]
        train_labels_path = temp_dir / files["train_labels"]
        
        if not train_images_path.exists():
            download_file(base_url + files["train_images"], train_images_path)
        if not train_labels_path.exists():
            download_file(base_url + files["train_labels"], train_labels_path)
        
        # Parse MNIST files
        print("Parsing MNIST data...")
        images = parse_mnist_images(train_images_path)
        labels = parse_mnist_labels(train_labels_path)
        
        # Take only the requested number
        images = images[:num_images]
        labels = labels[:num_images]
        
        # Save images
        print(f"Saving {num_images} images...")
        for idx, (image, label) in enumerate(zip(images, labels)):
            # Convert to PIL Image
            img = Image.fromarray(image, mode='L')
            
            # Save with label in filename: mnist_0001_digit5.png
            filename = f"mnist_{idx:04d}_digit{label}.png"
            filepath = output_path / filename
            img.save(filepath)
            
            if (idx + 1) % 100 == 0:
                print(f"  Saved {idx + 1}/{num_images} images...")
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"✓ Successfully saved {num_images} images to {output_dir}")
        
    except Exception as e:
        print(f"Error downloading MNIST dataset: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    # Set output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "mnist" / "images"
    
    # Download images
    success = download_mnist_images(str(output_dir), num_images=500)
    
    if success:
        print(f"\nImages are ready in: {output_dir}")
        print(f"Total files: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
