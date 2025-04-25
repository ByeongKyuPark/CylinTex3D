# Object Panorama Generation

This project implements a streamlined workflow for creating panoramas and 3D models from object images.

## Workflow

The pipeline consists of four main processes:

1. **Photos → Segmentation**: Extract objects from input photos
2. **Segmented object images → Panorama**: Create panoramas from the segmented objects
3. **Segmented object images → 3D Model**: Generate 3D models from segmented images, with plane removal
4. **3D Model → Texturing**: Apply textures to the 3D models

## Requirements

See `requirements.txt` for the necessary packages.

## Usage

Run the main script with:

```bash
python main.py --input_dir <path_to_input_images> --output_dir <path_to_output_directory>
```

### Optional arguments:

- `--skip_steps`: Skip specific steps in the pipeline. Options: segmentation, panorama, modeling, texturing.

Example:
```bash
python main.py --input_dir ./images --output_dir ./output --skip_steps segmentation
```

## Output Structure

The output directory will contain the following subdirectories:

- `segmented/`: Contains the segmented object images
- `panorama/`: Contains the generated panorama image
- `model/`: Contains the 3D model files
- `textured/`: Contains the textured 3D model and texture files
