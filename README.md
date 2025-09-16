# Document Scanner

A professional Python project for scanning and processing documents and images. This tool allows you to easily scan, enhance, and extract content from images or PDF files, making it ideal for digitizing paperwork, receipts, and other documents.

## Features

- Scan and process images or documents
- Automatic edge detection and perspective correction
- Image enhancement for better readability
- Batch processing support
- Output results to a dedicated folder

## Requirements

- Python 3.7+
- All dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MistaaOlivaaa/Document_Scanner.git
   cd excel
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Scan a Document

Run the main script to scan and process a document:

```bash
python demo_scanner.py --input /path/to/image.jpg --output ./output/
```

- `--input`: Path to the input image or document
- `--output`: Directory to save the processed output

### Example

```bash
python demo_scanner.py sample.jpg 
```

The processed document will be saved in the `output/` directory.

## Project Structure

- `demo_scanner.py` – Example script to demonstrate scanning
- `document_scanner.py` – Core scanning and processing logic
- `requirements.txt` – Python dependencies


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.