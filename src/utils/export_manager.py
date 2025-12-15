"""Export manager for exporting labeled images in various formats."""

import json
import csv
import shutil
from pathlib import Path
from typing import Dict
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ExportManager:
    """Manages exporting labeled images in various formats."""
    
    def __init__(self, parent=None):
        self.parent = parent
    
    def export_to_json(self, labeled_images: Dict[str, str], current_folder: Path = None):
        """
        Export labeled images to a JSON file.
        
        Args:
            labeled_images: Dictionary mapping image paths to labels
            current_folder: Current image folder for relative paths
        """
        print(f"DEBUG: ExportManager - Exporting {len(labeled_images)} images to JSON")
        
        if not labeled_images:
            QMessageBox.warning(
                self.parent,
                "No Labels",
                "There are no labeled images to export."
            )
            return
        
        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Export Labels to JSON",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            print("DEBUG: ExportManager - Export cancelled")
            return
        
        try:
            # Prepare data for export
            export_data = {
                "labels": [],
                "summary": {}
            }
            
            # Count labels and prepare entries
            label_counts = {}
            for img_path_str, label in labeled_images.items():
                img_path = Path(img_path_str)
                # Use forward slashes for cross-platform compatibility
                normalized_path = img_path.as_posix()
                export_data["labels"].append({
                    "filename": img_path.name,
                    "full_path": normalized_path,
                    "label": label
                })
                label_counts[label] = label_counts.get(label, 0) + 1
            
            export_data["summary"] = {
                "total_images": len(labeled_images),
                "label_counts": label_counts
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"DEBUG: ExportManager - Successfully exported to {file_path}")
            QMessageBox.information(
                self.parent,
                "Export Successful",
                f"Exported {len(labeled_images)} labeled image(s) to JSON file."
            )
        except Exception as e:
            print(f"DEBUG: ExportManager - Export failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Export Failed",
                f"Failed to export to JSON:\n{str(e)}"
            )
    
    def export_to_csv(self, labeled_images: Dict[str, str], current_folder: Path = None):
        """
        Export labeled images to a CSV file.
        
        Args:
            labeled_images: Dictionary mapping image paths to labels
            current_folder: Current image folder for relative paths
        """
        print(f"DEBUG: ExportManager - Exporting {len(labeled_images)} images to CSV")
        
        if not labeled_images:
            QMessageBox.warning(
                self.parent,
                "No Labels",
                "There are no labeled images to export."
            )
            return
        
        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Export Labels to CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            print("DEBUG: ExportManager - Export cancelled")
            return
        
        try:
            # Write to CSV file
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "full_path", "label"])
                
                for img_path_str, label in sorted(labeled_images.items()):
                    img_path = Path(img_path_str)
                    # Use forward slashes for cross-platform compatibility
                    normalized_path = img_path.as_posix()
                    writer.writerow([img_path.name, normalized_path, label])
            
            print(f"DEBUG: ExportManager - Successfully exported to {file_path}")
            QMessageBox.information(
                self.parent,
                "Export Successful",
                f"Exported {len(labeled_images)} labeled image(s) to CSV file."
            )
        except Exception as e:
            print(f"DEBUG: ExportManager - Export failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Export Failed",
                f"Failed to export to CSV:\n{str(e)}"
            )
    
    def export_organized_copy(self, labeled_images: Dict[str, str]):
        """
        Copy labeled images to a new directory organized by label subfolders.
        
        Args:
            labeled_images: Dictionary mapping image paths to labels
        """
        print(f"DEBUG: ExportManager - Organizing {len(labeled_images)} images into folders")
        
        if not labeled_images:
            QMessageBox.warning(
                self.parent,
                "No Labels",
                "There are no labeled images to export."
            )
            return
        
        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self.parent,
            "Select Output Directory for Organized Images",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not output_dir:
            print("DEBUG: ExportManager - Export cancelled")
            return
        
        try:
            output_path = Path(output_dir)
            copied_count = 0
            label_counts = {}
            
            # Copy images to organized folders
            for img_path_str, label in labeled_images.items():
                img_path = Path(img_path_str)
                
                # Create label subfolder if it doesn't exist
                label_folder = output_path / label
                label_folder.mkdir(parents=True, exist_ok=True)
                
                # Copy image to label folder
                dest_path = label_folder / img_path.name
                shutil.copy2(img_path, dest_path)
                
                copied_count += 1
                label_counts[label] = label_counts.get(label, 0) + 1
                print(f"DEBUG: ExportManager - Copied {img_path.name} to {label}/")
            
            # Create summary
            summary_lines = [f"{label}: {count} image(s)" for label, count in sorted(label_counts.items())]
            summary_text = "\n".join(summary_lines)
            
            print(f"DEBUG: ExportManager - Successfully copied {copied_count} images")
            QMessageBox.information(
                self.parent,
                "Export Successful",
                f"Copied {copied_count} labeled image(s) to:\n{output_path}\n\nOrganized by label:\n{summary_text}"
            )
        except Exception as e:
            print(f"DEBUG: ExportManager - Export failed: {e}")
            QMessageBox.critical(
                self.parent,
                "Export Failed",
                f"Failed to organize and copy images:\n{str(e)}"
            )
