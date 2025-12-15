"""Model for managing a collection of images."""

from pathlib import Path
from typing import List, Set


class ImageCollection:
    """Manages a collection of image file paths."""
    
    # Supported image extensions
    IMAGE_EXTENSIONS: Set[str] = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    def __init__(self):
        self.image_paths: List[Path] = []
        self.current_folder: Path | None = None
    
    def load_from_folder(self, folder_path: Path) -> int:
        """
        Load all images from a folder.
        
        Args:
            folder_path: Path to folder to scan
            
        Returns:
            Number of images found
        """
        self.current_folder = folder_path
        self.image_paths = []
        
        # Scan for image files
        for file_path in sorted(folder_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                self.image_paths.append(file_path)
        
        return len(self.image_paths)
    
    def get_images(self) -> List[Path]:
        """Get list of all image paths."""
        return self.image_paths
    
    def get_folder(self) -> Path | None:
        """Get current folder path."""
        return self.current_folder
    
    def clear(self):
        """Clear the collection."""
        self.image_paths = []
        self.current_folder = None
