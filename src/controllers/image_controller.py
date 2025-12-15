"""Controller for managing image operations."""

from pathlib import Path
from src.models.image_collection import ImageCollection


class ImageController:
    """Handles business logic for image operations."""
    
    def __init__(self):
        self.collection = ImageCollection()
    
    def load_folder(self, folder_path: str | Path) -> tuple[int, Path]:
        """
        Load images from a folder.
        
        Args:
            folder_path: Path to folder
            
        Returns:
            Tuple of (number of images loaded, folder path)
        """
        folder = Path(folder_path)
        count = self.collection.load_from_folder(folder)
        print(f"Loaded {count} images from {folder}")
        return count, folder
    
    def get_images(self):
        """Get list of image paths."""
        return self.collection.get_images()
    
    def get_current_folder(self):
        """Get current folder path."""
        return self.collection.get_folder()
