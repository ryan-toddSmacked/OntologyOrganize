# Classifier Organizer 2

A PyQt5 application for organizing and classifying data.

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup Test Data

Download 500 MNIST test images:
```bash
python scripts/downloadTestDataset.py
```

This will download images to `data/mnist/images/`. No extra dependencies needed!

## Running the Application

From the project root:
```bash
python -m src.main
```

Or from the root-level main.py:
```bash
python main.py
```

## Project Structure

```
src/
├── main.py              # Application entry point
├── ui/                  # UI components
│   ├── main_window.py   # Main application window
│   ├── widgets/         # Custom widgets
│   └── dialogs/         # Dialog windows
├── models/              # Data models
├── controllers/         # Business logic
├── utils/               # Helper functions
└── resources/           # Images, icons, styles
```

## Development

- Add new UI components in `src/ui/`
- Add data models in `src/models/`
- Add business logic in `src/controllers/`
- Add utility functions in `src/utils/`
- Add tests in `tests/`
