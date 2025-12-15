"""Progress manager for saving and loading application state."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ProgressManager:
    """Manages saving and loading application progress."""
    
    def __init__(self, parent=None):
        self.parent = parent
    
    def save_progress(self, state_data: Dict[str, Any]):
        """
        Save current application state to a JSON file.
        
        Args:
            state_data: Dictionary containing all application state
        """
        print("DEBUG: ProgressManager - Saving progress")
        
        # Generate default filename with UTC timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        default_filename = f"classifier_progress_{timestamp}.json"
        
        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Save Progress",
            default_filename,
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            print("DEBUG: ProgressManager - Save cancelled")
            return False
        
        try:
            # Add metadata
            save_data = {
                "metadata": {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0"
                },
                "state": state_data
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            
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
    
    def load_progress(self) -> Dict[str, Any]:
        """
        Load application state from a JSON file.
        
        Returns:
            Dictionary containing application state, or None if cancelled/failed
        """
        print("DEBUG: ProgressManager - Loading progress")
        
        # Ask user for file to load
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Load Progress",
            "",
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
            if "state" not in loaded_data:
                raise ValueError("Invalid progress file format - missing 'state' key")
            
            state_data = loaded_data["state"]
            
            # Log metadata if present
            if "metadata" in loaded_data:
                metadata = loaded_data["metadata"]
                print(f"DEBUG: ProgressManager - Loaded file saved at: {metadata.get('saved_at', 'unknown')}")
                print(f"DEBUG: ProgressManager - File version: {metadata.get('version', 'unknown')}")
            
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
        except Exception as e:
            print(f"DEBUG: ProgressManager - Load failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Load Failed",
                f"Failed to load progress:\n{str(e)}"
            )
            return None
