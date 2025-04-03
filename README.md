# GoColonyCounter

GoColonyCounter is a lightweight CLI tool to automatically count bacterial colonies in Petri dish images using HSV brightness thresholding and watershed-based blob detection.

- Detects colonies from images of Petri dishes
- Supports JPEG, PNG, TIFF formats (including iPhone photos)
- Option to split plates into left/right halves
- Exports result images and CSV count tables

## Author

Heekuk Park  
Email: hp2523@cumc.columbia.edu  
Version: v1.02  
Date: 2025-04-02

## Installation

To install the required Python packages and make the command globally accessible:

1. Clone the repository and move into the directory:
    ```bash
    git clone https://github.com/bbagy/GoColonyCounter.git
    cd GoColonyCounter
    ```

2. Run the installation script:
    ```bash
    bash install.sh
    ```

This will:
- Install required Python packages (opencv-python, numpy, scipy)
- Grant execute permission to GoColonyCounter.py
- Create a symbolic link at /usr/local/bin/GoColonyCounter

After installation, you can run the tool from anywhere using:
```bash
GoColonyCounter -i input_images -o output_results
```

## Usage

```bash
GoColonyCounter -i input_images -o output_results
```

### Optional flags

| Flag | Description | Example |
|------|-------------|---------|
| `-p` | Plate mode: `1` = full plate, `2` = split left/right | `-p 2` |
| `--min_area` | Minimum area for a detected colony | `--min_area 0` |
| `--brightness_threshold` | Brightness threshold (HSV V channel, 0–255) | `--brightness_threshold 185` |

## Output

- `colony_counts.csv` — summary table of colony counts per image  
- `result_*.png` — annotated result images with colony markers and counts

## Notes

- Works on macOS (recommended); Linux also supported
- iPhone .jpg images are supported (HEIC must be converted)
- Resize is automatic (resizes to ~800px max side)

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software for any purpose, including commercial applications, provided that the original author is credited and the license terms are preserved.

See the [LICENSE](LICENSE) file for full details.
