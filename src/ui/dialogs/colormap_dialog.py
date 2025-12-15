"""Dialog for selecting colormap."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QGroupBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt


class ColormapDialog(QDialog):
    """Dialog for selecting colormap."""
    
    def __init__(self, parent=None, colormap="gray"):
        super().__init__(parent)
        self.colormap = colormap
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Colormap Settings")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Colormap settings group
        colormap_group = QGroupBox("Select Colormap")
        colormap_layout = QFormLayout()
        colormap_group.setLayout(colormap_layout)
        
        # Colormap selector
        self.colormap_combo = QComboBox()
        colormaps = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "batlow",
            "berlin",
            "broc",
            "cork",
            "hawaii",
            "imola",
            "lajolla",
            "lapaz",
            "nuuk",
            "oslo",
            "roma",
            "tokyo",
            "turku",
            "vik"
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText(self.colormap)
        colormap_layout.addRow("Colormap:", self.colormap_combo)
        
        layout.addWidget(colormap_group)
        layout.addStretch()
        
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
    
    def get_colormap(self):
        """Get the selected colormap."""
        return self.colormap_combo.currentText()
