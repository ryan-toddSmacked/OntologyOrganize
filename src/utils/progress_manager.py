"""Progress manager for saving and loading application state."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ProgressManager:
    """Manages saving and loading application progress."""
    
    CURRENT_VERSION = "1.1"
    
    def __init__(self, parent=None):
        self.parent = parent
        self.last_save_path: Optional[str] = None
        self.last_load_path: Optional[str] = None
    
    def save_progress(self, state_data: Dict[str, Any], file_path: Optional[str] = None) -> bool:
        """
        Save current application state to a JSON file.
        
        Args:
            state_data: Dictionary containing all application state
            file_path: Optional path to save to (if None, prompts user)
        
        Returns:
            True if save was successful, False otherwise
        """
        print("DEBUG: ProgressManager - Saving progress")
        
        if file_path is None:
            # Generate default filename with UTC timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
            default_filename = f"classifier_progress_{timestamp}.json"
            
            # Use last save directory if available
            default_path = default_filename
            if self.last_save_path:
                last_dir = str(Path(self.last_save_path).parent)
                default_path = str(Path(last_dir) / default_filename)
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Save Progress",
                default_path,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                print("DEBUG: ProgressManager - Save cancelled")
                return False
        
        try:
            # Validate state_data before saving
            if not isinstance(state_data, dict):
                raise ValueError("State data must be a dictionary")
            
            # Add metadata
            save_data = {
                "metadata": {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "version": self.CURRENT_VERSION,
                    "app_name": "OntologyOrganize",
                    "image_count": len(state_data.get("labeled_images", {}))
                },
                "state": state_data
            }
            
            # Write to file with atomic operation
            temp_path = f"{file_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            
            # Rename to final path (atomic on most systems)
            Path(temp_path).replace(file_path)
            
            # Remember this path for next time
            self.last_save_path = file_path
            
            print(f"DEBUG: ProgressManager - Successfully saved to {file_path}")
            QMessageBox.information(
                self.parent,
                "Save Successful",
                f"Progress saved successfully to:\n{file_path}"
            )
            return True
        except Exception as e:
            print(f"DEBUG: ProgressManager - Save failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Save Failed",
                f"Failed to save progress:\n{str(e)}"
            )
            return False
    
    def load_progress(self, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load application state from a JSON file.
        
        Args:
            file_path: Optional path to load from (if None, prompts user)
        
        Returns:
            Dictionary containing application state, or None if cancelled/failed
        """
        print("DEBUG: ProgressManager - Loading progress")
        
        if file_path is None:
            # Use last load directory if available
            default_path = ""
            if self.last_load_path:
                default_path = str(Path(self.last_load_path).parent)
            
            # Ask user for file to load
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent,
                "Load Progress",
                default_path,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                print("DEBUG: ProgressManager - Load cancelled")
                return None
        
        try:
            # Read from file
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # Validate structure
            if not isinstance(loaded_data, dict):
                raise ValueError("Invalid progress file format - expected a dictionary")
            
            if "state" not in loaded_data:
                raise ValueError("Invalid progress file format - missing 'state' key")
            
            state_data = loaded_data["state"]
            
            # Validate state data
            if not isinstance(state_data, dict):
                raise ValueError("Invalid state data - expected a dictionary")
            
            # Log and validate metadata
            metadata = loaded_data.get("metadata", {})
            if metadata:
                saved_at = metadata.get('saved_at', 'unknown')
                version = metadata.get('version', 'unknown')
                image_count = metadata.get('image_count', 'unknown')
                print(f"DEBUG: ProgressManager - Loaded file saved at: {saved_at}")
                print(f"DEBUG: ProgressManager - File version: {version}")
                print(f"DEBUG: ProgressManager - Image count: {image_count}")
                
                # Version compatibility check
                if version != 'unknown' and version < "1.0":
                    print(f"WARNING: Loading older version file ({version})")
            
            # Remember this path for next time
            self.last_load_path = file_path
            
            print(f"DEBUG: ProgressManager - Successfully loaded from {file_path}")
            QMessageBox.information(
                self.parent,
                "Load Successful",
                f"Progress loaded successfully from:\n{file_path}"
            )
            return state_data
        except json.JSONDecodeError as e:
            print(f"DEBUG: ProgressManager - Invalid JSON: {e}")
            QMessageBox.critical(
                self.parent,
                "Load Failed",
                f"Failed to load progress - invalid JSON file:\n{str(e)}"
            )
            return None
        except FileNotFoundError:
            print(f"ERROR: ProgressManager - File not found: {file_path}")
            QMessageBox.critical(
                self.parent,
                "Load Failed",
                f"File not found:\n{file_path}"
            )
            return None
        except ValueError as e:
            print(f"ERROR: ProgressManager - Validation error: {e}")
            QMessageBox.critical(
                self.parent,
                "Load Failed",
                f"Invalid progress file:\n{str(e)}"
            )
            return None
        except Exception as e:
            print(f"DEBUG: ProgressManager - Load failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Load Failed",
                f"Failed to load progress:\n{str(e)}"
            )
            return None
    
    def quick_save(self, state_data: Dict[str, Any]) -> bool:
        """
        Quick save to the last used file path without prompting.
        
        Args:
            state_data: Dictionary containing all application state
        
        Returns:
            True if save was successful, False otherwise
        """
        if self.last_save_path:
            return self.save_progress(state_data, self.last_save_path)
        else:
            # No previous save path, do regular save
            return self.save_progress(state_data)
    
    def get_recent_files(self) -> Dict[str, str]:
        """
        Get the most recently used save and load paths.
        
        Returns:
            Dictionary with 'save' and 'load' keys containing file paths
        """
        return {
            "save": self.last_save_path,
            "load": self.last_load_path
        }
