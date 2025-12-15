"""View manager for controlling what images are displayed in the grid."""

from enum import Enum
from typing import List
from pathlib import Path


class ViewMode(Enum):
    """Enum for different view modes."""
    UNLABELED = "unlabeled"
    LABELED = "labeled"
    ALL = "all"


class ViewManager:
    """Manages the view mode and filtering of images."""
    
    def __init__(self):
        self.view_mode = ViewMode.UNLABELED
    
    def set_view_mode(self, mode: ViewMode):
        """Set the current view mode."""
        print(f"DEBUG: ViewManager - Setting view mode to {mode.value}")
        self.view_mode = mode
    
    def get_view_mode(self) -> ViewMode:
        """Get the current view mode."""
        return self.view_mode
    
    def get_current_view_mode(self) -> ViewMode:
        """Get the current view mode (alias for compatibility)."""
        return self.view_mode
    
    def filter_images(self, all_images: List[Path], labeled_images: dict) -> List[Path]:
        """
        Filter images based on the current view mode.
        
        Args:
            all_images: List of all image paths
            labeled_images: Dictionary mapping image paths to labels
            
        Returns:
            Filtered list of image paths based on view mode
        """
        print(f"DEBUG: ViewManager - Filtering {len(all_images)} images in {self.view_mode.value} mode")
        
        if self.view_mode == ViewMode.UNLABELED:
            # Show only unlabeled images
            filtered = [img for img in all_images if str(img) not in labeled_images]
            print(f"DEBUG: ViewManager - Filtered to {len(filtered)} unlabeled images")
            return filtered
        
        elif self.view_mode == ViewMode.LABELED:
            # Show only labeled images
            filtered = [img for img in all_images if str(img) in labeled_images]
            print(f"DEBUG: ViewManager - Filtered to {len(filtered)} labeled images")
            return filtered
        
        elif self.view_mode == ViewMode.ALL:
            # Show all images
            print(f"DEBUG: ViewManager - Showing all {len(all_images)} images")
            return all_images
        
        return all_images
    
    def get_mode_name(self) -> str:
        """Get a human-readable name for the current mode."""
        mode_names = {
            ViewMode.UNLABELED: "Unlabeled Images",
            ViewMode.LABELED: "Labeled Images",
            ViewMode.ALL: "All Images"
        }
        return mode_names.get(self.view_mode, "Unknown")
