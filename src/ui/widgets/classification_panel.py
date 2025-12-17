"""Classification panel widget for ontology management."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QFileDialog, QInputDialog, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt
from pathlib import Path


class ClassificationPanel(QWidget):
    """Widget for managing ontology labels and classification."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Store reference to main window
        self.ontology_labels = []  # List of ontology labels
        self.ontology_file_path = None  # Path to loaded ontology file
        self.binary_mode = False  # Whether in binary classification mode
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Classification header
        header_label = QLabel("Classification Area")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header_label)
        
        # Ontology file controls
        ontology_controls_layout = QHBoxLayout()
        self.load_ontology_btn = QPushButton("Load Ontology...")
        self.load_ontology_btn.clicked.connect(self.load_ontology_file)
        ontology_controls_layout.addWidget(self.load_ontology_btn)
        
        self.save_ontology_btn = QPushButton("Save Ontology...")
        self.save_ontology_btn.clicked.connect(self.save_ontology_file)
        ontology_controls_layout.addWidget(self.save_ontology_btn)
        
        layout.addLayout(ontology_controls_layout)
        
        # Ontology file path label
        self.ontology_file_label = QLabel("No ontology loaded")
        self.ontology_file_label.setStyleSheet("font-size: 10px; color: gray;")
        self.ontology_file_label.setWordWrap(True)
        layout.addWidget(self.ontology_file_label)
        
        # Labels list
        labels_header = QLabel("Ontology Labels:")
        labels_header.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(labels_header)
        
        self.ontology_list = QListWidget()
        self.ontology_list.setStyleSheet("""
            QListWidget {
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 10px;
                margin: 2px;
                border: 2px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            QListWidget::item:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
                border-color: #005a9e;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.ontology_list)
        
        # Binary mode widgets (initially hidden)
        self.binary_widget = QWidget()
        binary_layout = QVBoxLayout()
        self.binary_widget.setLayout(binary_layout)
        
        # Binary mode label list
        binary_header = QLabel("Binary Classes:")
        binary_header.setStyleSheet("font-weight: bold;")
        binary_layout.addWidget(binary_header)
        
        self.binary_list = QListWidget()
        self.binary_list.setStyleSheet("""
            QListWidget {
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 10px;
                margin: 2px;
                border: 2px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            QListWidget::item:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
                border-color: #005a9e;
                font-weight: bold;
            }
        """)
        self.binary_list.currentItemChanged.connect(self.update_selected_label_display)
        binary_layout.addWidget(self.binary_list)
        
        # Add instruction for binary mode
        binary_instruction = QLabel("Select a class above, then click images and label them below.\nEdit class names in Settings > Binary Mode.")
        binary_instruction.setStyleSheet("font-size: 10px; color: #666; margin-top: 5px;")
        binary_instruction.setWordWrap(True)
        binary_layout.addWidget(binary_instruction)
        
        layout.addWidget(self.binary_widget)
        self.binary_widget.hide()  # Hidden by default
        
        # Add/Remove label buttons
        label_controls_layout = QHBoxLayout()
        
        self.add_label_btn = QPushButton("Add Label")
        self.add_label_btn.clicked.connect(self.add_ontology_label)
        label_controls_layout.addWidget(self.add_label_btn)
        
        self.remove_label_btn = QPushButton("Remove Label")
        self.remove_label_btn.clicked.connect(self.remove_ontology_label)
        label_controls_layout.addWidget(self.remove_label_btn)
        
        layout.addLayout(label_controls_layout)
        
        # Instructions label
        instructions_label = QLabel("1. Select a label below\n2. Click images in grid\n3. Click button to apply label")
        instructions_label.setStyleSheet("font-size: 10px; color: #666; margin-top: 10px; margin-bottom: 5px;")
        instructions_label.setWordWrap(True)
        layout.addWidget(instructions_label)
        
        # Currently selected label display
        self.current_label_display = QLabel("No label selected")
        self.current_label_display.setStyleSheet(
            "background-color: #f0f0f0; padding: 8px; border: 2px solid #ccc; "
            "font-weight: bold; font-size: 11px; color: #333;"
        )
        self.current_label_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.current_label_display)
        
        # Label selected images button
        self.label_images_btn = QPushButton("Label Selected Images")
        self.label_images_btn.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; padding: 10px; font-size: 12px;"
        )
        self.label_images_btn.clicked.connect(self.label_selected_images)
        layout.addWidget(self.label_images_btn)
        
        # Connect to update display when selection changes
        self.ontology_list.currentItemChanged.connect(self.update_selected_label_display)
    
    def load_ontology_file(self):
        """Load ontology labels from a newline-separated file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Ontology File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse labels (strip whitespace and ignore empty lines)
                labels = [line.strip() for line in lines if line.strip()]
                
                if labels:
                    self.ontology_labels = labels
                    self.ontology_file_path = file_path
                    self.update_ontology_list()
                    self.ontology_file_label.setText(f"Loaded: {Path(file_path).name}")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Loaded {len(labels)} label(s) from ontology file."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Empty File",
                        "The selected file contains no valid labels."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load ontology file:\n{str(e)}"
                )
    
    def save_ontology_file(self):
        """Save current ontology labels to a newline-separated file."""
        if not self.ontology_labels:
            QMessageBox.warning(
                self,
                "No Labels",
                "There are no labels to save."
            )
            return
        
        # Use existing file path as default if available
        default_path = self.ontology_file_path if self.ontology_file_path else ""
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Ontology File",
            default_path,
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for label in self.ontology_labels:
                        f.write(f"{label}\n")
                
                self.ontology_file_path = file_path
                self.ontology_file_label.setText(f"Loaded: {Path(file_path).name}")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Saved {len(self.ontology_labels)} label(s) to ontology file."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save ontology file:\n{str(e)}"
                )
    
    def add_ontology_label(self):
        """Add new labels to the ontology (supports multiple labels, one per line)."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Add Ontology Labels",
            "Enter label names (one per line):"
        )
        
        if ok and text:
            # Parse labels (strip whitespace and ignore empty lines)
            new_labels = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not new_labels:
                return
            
            # Track duplicates and added labels
            duplicates = []
            added = []
            
            for label in new_labels:
                if label in self.ontology_labels:
                    duplicates.append(label)
                else:
                    self.ontology_labels.append(label)
                    added.append(label)
            
            # Update the list if any labels were added
            if added:
                self.update_ontology_list()
            
            # Show appropriate message
            if added and duplicates:
                QMessageBox.information(
                    self,
                    "Labels Added",
                    f"Added {len(added)} label(s).\n\n"
                    f"{len(duplicates)} duplicate(s) were skipped:\n{', '.join(duplicates)}"
                )
            elif added:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Added {len(added)} label(s) to the ontology."
                )
            elif duplicates:
                QMessageBox.warning(
                    self,
                    "Duplicate Labels",
                    f"All {len(duplicates)} label(s) already exist:\n{', '.join(duplicates)}"
                )
    
    def remove_ontology_label(self):
        """Remove selected label from the ontology."""
        current_item = self.ontology_list.currentItem()
        
        if current_item:
            label = current_item.text()
            reply = QMessageBox.question(
                self,
                "Remove Label",
                f"Are you sure you want to remove the label '{label}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.ontology_labels.remove(label)
                self.update_ontology_list()
        else:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a label to remove."
            )
    
    def update_ontology_list(self):
        """Update the ontology labels list widget."""
        self.ontology_list.clear()
        for label in self.ontology_labels:
            self.ontology_list.addItem(label)
    
    def get_labels(self):
        """Get the list of ontology labels."""
        return self.ontology_labels.copy()
    
    def set_labels(self, labels):
        """Set the ontology labels."""
        self.ontology_labels = labels.copy()
        self.update_ontology_list()
    
    def label_selected_images(self):
        """Emit signal or call parent to label selected images."""
        print("DEBUG: label_selected_images called")
        
        # Get current item from the appropriate list
        if self.binary_mode:
            current_item = self.binary_list.currentItem()
            list_widget = self.binary_list
        else:
            current_item = self.ontology_list.currentItem()
            list_widget = self.ontology_list
        
        print(f"DEBUG: Current ontology list item count: {list_widget.count()}")
        print(f"DEBUG: Current row selected: {list_widget.currentRow()}")
        print(f"DEBUG: Current item object: {current_item}")
        if not current_item:
            print("DEBUG: No label selected")
            QMessageBox.warning(
                self,
                "No Label Selected",
                "Please select a label from the ontology list first."
            )
            return
        
        label = current_item.text()
        print(f"DEBUG: Selected label text: '{label}'")
        # Call main window's method to handle labeling
        print(f"DEBUG: self.main_window = {self.main_window}")
        print(f"DEBUG: hasattr check = {hasattr(self.main_window, 'label_selected_images_with_label') if self.main_window else 'N/A'}")
        if self.main_window and hasattr(self.main_window, 'label_selected_images_with_label'):
            print(f"DEBUG: Calling main_window's label_selected_images_with_label with label={label}")
            self.main_window.label_selected_images_with_label(label)
        else:
            print(f"DEBUG: Cannot call main_window method")
    
    def update_selected_label_display(self):
        """Update the display showing which label is currently selected."""
        # Get current item from the appropriate list
        if self.binary_mode:
            current_item = self.binary_list.currentItem()
        else:
            current_item = self.ontology_list.currentItem()
        
        if current_item:
            label = current_item.text()
            self.current_label_display.setText(f"Active Label: {label}")
            self.current_label_display.setStyleSheet(
                "background-color: #d4edda; padding: 8px; border: 2px solid #28a745; "
                "font-weight: bold; font-size: 11px; color: #155724;"
            )
            # Notify main window of label change
            if self.main_window and hasattr(self.main_window, 'set_active_label'):
                self.main_window.set_active_label(label)
        else:
            self.current_label_display.setText("No label selected")
            self.current_label_display.setStyleSheet(
                "background-color: #f0f0f0; padding: 8px; border: 2px solid #ccc; "
                "font-weight: bold; font-size: 11px; color: #333;"
            )
            # Notify main window of no label
            if self.main_window and hasattr(self.main_window, 'set_active_label'):
                self.main_window.set_active_label(None)
    
    def get_selected_label(self):
        """Get the currently selected label."""
        if self.binary_mode:
            current_item = self.binary_list.currentItem()
        else:
            current_item = self.ontology_list.currentItem()
        
        if current_item:
            return current_item.text()
        return None
    
    def get_ontology_labels(self):
        """Get the list of ontology labels."""
        return self.ontology_labels.copy()
    
    def set_ontology_labels(self, labels, label_colors=None):
        """
        Set the ontology labels and optionally restore label colors.
        
        Args:
            labels: List of label strings
            label_colors: Optional dict mapping labels to colors
        """
        print(f"DEBUG: Setting ontology labels: {labels}")
        self.ontology_labels = labels.copy()
        
        # Update the list widget
        self.ontology_list.clear()
        for label in self.ontology_labels:
            self.ontology_list.addItem(label)
        
        # Restore label colors if provided
        if label_colors and self.main_window:
            for label, color in label_colors.items():
                if label in self.ontology_labels:
                    self.main_window.label_colors[label] = color
        
        print(f"DEBUG: Ontology labels set successfully")
    
    def set_binary_mode(self, enabled, positive_class="Positive Class", negative_class="Negative Class"):
        """
        Enable or disable binary classification mode.
        
        Args:
            enabled: True to enable binary mode, False for multi-label mode
            positive_class: Label for positive class
            negative_class: Label for negative class
        """
        self.binary_mode = enabled
        
        if enabled:
            # Show binary mode widgets, hide multi-label widgets
            self.ontology_list.hide()
            self.add_label_btn.hide()
            self.remove_label_btn.hide()
            self.load_ontology_btn.setEnabled(False)
            self.save_ontology_btn.setEnabled(False)
            self.binary_widget.show()
            
            # Update the internal ontology labels to match binary mode
            self.ontology_labels = [positive_class, negative_class]
            
            # Populate binary list
            self.binary_list.clear()
            self.binary_list.addItem(positive_class)
            self.binary_list.addItem(negative_class)
            
            # Also update the hidden ontology list for consistency
            self.update_ontology_list()
            
            # Select the first label (positive) by default
            if self.binary_list.count() > 0:
                self.binary_list.setCurrentRow(0)
        else:
            # Show multi-label widgets, hide binary mode widgets
            self.ontology_list.show()
            self.add_label_btn.show()
            self.remove_label_btn.show()
            self.load_ontology_btn.setEnabled(True)
            self.save_ontology_btn.setEnabled(True)
            self.binary_widget.hide()
    
    def get_binary_classes(self):
        """Get the current binary class labels."""
        if self.binary_list.count() >= 2:
            return (self.binary_list.item(0).text(), 
                    self.binary_list.item(1).text())
        return ("Positive Class", "Negative Class")
