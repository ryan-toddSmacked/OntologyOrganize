"""Settings dialog for configuring application preferences."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QPushButton, QGroupBox, QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt


class SettingsDialog(QDialog):
    """Dialog for application settings."""
    
    def __init__(self, parent=None, grid_cols=3, grid_rows=4):
        super().__init__(parent)
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(300, 200)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Grid settings group
        grid_group = QGroupBox("Image Grid")
        grid_layout = QFormLayout()
        grid_group.setLayout(grid_layout)
        
        # Columns spinner
        self.cols_spinner = QSpinBox()
        self.cols_spinner.setMinimum(1)
        self.cols_spinner.setMaximum(20)
        self.cols_spinner.setValue(self.grid_cols)
        grid_layout.addRow("Columns:", self.cols_spinner)
        
        # Rows spinner
        self.rows_spinner = QSpinBox()
        self.rows_spinner.setMinimum(1)
        self.rows_spinner.setMaximum(1000)
        self.rows_spinner.setValue(self.grid_rows)
        grid_layout.addRow("Rows:", self.rows_spinner)
        
        layout.addWidget(grid_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
    
    def get_grid_size(self):
        """Get the selected grid size."""
        return self.cols_spinner.value(), self.rows_spinner.value()
    
    def get_colormap(self):
        """Get the selected colormap."""
        return self.colormap_combo.currentText()
