<p align="center">
  <h1 align="center">Image Stitching using SuperGlue</h1>
  <p align="center">
    <img src="./data/results.png" alt="Panorama Image" width="1600">
  <p align="center">
    <a href="https://github.com/Factral/" rel="external nofollow noopener" target="_blank"><strong>Fabian Perez</strong></a>
    ·
    <a href="https://github.com/Factral/" rel="external nofollow noopener" target="_blank"><strong>Paula Arguello</strong></a>
    ·
    <a href="https://github.com/Factral/" rel="external nofollow noopener" target="_blank"><strong>Mariana Robayo</strong></a>
  </p>
<p align="center">
    Digital Image Processing project at UIS 2024-1

- Visit the  [report](report.pdf) for more information about the project.
- Visit the  [slides](slides.pdf) for the presentation.

## Overview

This repository contains a Python script for stitching images together using [SuperGlue network](https://github.com/magicleap/SuperGluePretrainedNetwork) for matching keypoints and OpenCV for image processing. The script processes a folder of images, aligning and blending them to create a seamless panorama.

### Arguments

The `stitching.py` script accepts the following arguments:

- `--folder`: Folder path containing images (default: `./acquisitions`)
- `--crop`: Boolean flag to crop the panorama to remove black borders (default: `False`)
- `--blend`: Boolean flag to blend the images (default: `False`)
- `--showsteps`: Boolean flag to show the steps of the panorama process (default: `False`)

### Image Order

Ensure that the images in the folder are ordered from left to right. It is recommended to name the images sequentially, e.g., `image1.jpg`, `image2.jpg`, etc. From the leftmost image to the rightmost image.

### Installation

To set up the environment for running the script, follow these steps:

```bash
git clone https://github.com/Factral/image-stitching-supeglue
cd image-stitching-supeglue
pip install -r requirements.txt
```

this code was developed with python 3.10

### Running the Script

To create a panorama from the images in the specified folder, use the following command:

```bash
python3 stitching.py --folder path/to/your/images --crop --blend --showsteps
```

you can play with the arguments to see the different results.

## Code Structure Explanation

The codebase is structured as follows:

- `stitching.py`: The main script for stitching images.
- `match_pairs.py`: A script used for matching keypoints between image pairs using SuperGlue.
- `requirements.txt`: A file listing the dependencies required to run the script.


## License

This project is licensed under the MIT License

feel free to use it and modify it as you wish, if you find this code useful please give a star ⭐ to the repository.
