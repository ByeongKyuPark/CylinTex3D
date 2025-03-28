# Cylindrical Object Panorama Creator

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Input Image Sample</b></td>
      <td align="center"><b>Generated Panorama Result</b></td>
    </tr>
    <tr>
      <td><img src="docs/sample_input.jpg" width="300" alt="Sample Input Image"/></td>
      <td><img src="docs/panorama_result.jpg" width="600" alt="Panorama Result"/></td>
    </tr>
  </table>
</div>

## Overview

This tool creates seamless panoramas from images of cylindrical objects by automatically stitching together photos taken while rotating around the object. Unlike traditional panorama tools, this project is specifically designed for cylindrical objects where distortion presents a unique challenge.

The tool uses a flexible parameter approach that adapts to the specific characteristics of each image pair, automatically finding the optimal alignment parameters for each stitch. It also supports vertical adjustments to account for camera alignment variations.

## Features

- **Object Segmentation**: Automatically extracts the object from the background using the GrabCut algorithm
- **Flexible Parameter Matching**: Tests multiple parameters for each stitch and selects the best combination
- **Vertical Offset Support**: Accounts for slight camera height variations between shots
- **Intelligent Blending**: Seamlessly combines strips to create a continuous panorama
- **Detailed Visualizations**: Generates step-by-step visualizations of the stitching process

## Requirements

- Python 3.6+
- OpenCV 4.5+
- NumPy
- Matplotlib
- tqdm

To install the required dependencies:

```bash
# Most reliable method (works on all systems)
python -m pip install -r requirements.txt

# Alternative methods if you encounter issues:
# Install packages individually
python -m pip install numpy
python -m pip install opencv-python
python -m pip install matplotlib
python -m pip install tqdm

# If using Anaconda
# conda install numpy opencv matplotlib tqdm
```

## Usage

### Basic Usage

```bash
python main.py --input_dir images
```

This will:
1. Segment objects from all images in the `images` directory
2. Create a panorama using optimal parameters for each stitch
3. Save the result to the `results` directory

### Custom Options

```bash
python main.py --input_dir images --output_dir my_results --match_widths 20,30,40 --y_offset_range -8,8
```

Run with `--help` to see all available options:

```bash
python main.py --help
```

### Skip Segmentation

If you already have segmented images:

```bash
python main.py --input_dir images --skip_segmentation
```

## How It Works

### 1. Object Segmentation

The tool first separates the object from the background using the GrabCut algorithm. This creates clean silhouettes that improve the matching process.

### 2. Strip Matching

For each pair of adjacent images, the algorithm:

1. Tests multiple combinations of strip widths and search regions
2. Tries different vertical offsets to account for camera alignment variations
3. Calculates the Sum of Squared Differences (SSD) for each combination
4. Selects the parameters that produce the lowest SSD (best match)

### 3. Panorama Creation

The panorama is built incrementally:

1. Starts with a center strip from the first image
2. For each subsequent image, finds the best matching position and extracts a strip
3. Adds each new strip to the left side of the current panorama
4. Applies alpha blending to smooth transitions between strips

## Project Structure

```
cylindrical-panorama/
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── main.py                # Main execution script
├── utils/
│   ├── image_utils.py     # Image loading and basic operations
│   └── visualization.py   # Visualization functions
├── segmentation/
│   └── grabcut.py         # Object segmentation code
└── panorama/
    ├── strip_matching.py  # Strip matching algorithms
    └── stitching.py       # Panorama creation
```

## Example Results

The tool generates several outputs:

- `panorama.png`: The final panorama
- `panorama_white.png`: The panorama with a white background
- `matching_*.png`: Visualization of each matching step
- `parameters_log.txt`: Log of parameters used for each image

## Limitations

- Images must be taken in a consistent order (clockwise or counterclockwise)
- The algorithm assumes the object is centered in each image
- Better results are achieved with overlapping photos (20-30% overlap recommended)
- Very reflective or transparent objects may produce inconsistent results

## License

[MIT License](LICENSE)

## Acknowledgments

- Algorithm based on strip-matching techniques for cylindrical objects
- Inspired by panorama stitching approaches that adapt to non-planar surfaces