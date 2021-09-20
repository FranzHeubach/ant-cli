# AUV Navigation Testbed Toolchain

![Gazebo World](images/gazebo-world.png)

This python package provides a few tools for working with rasters with regards to:
 * Create a Gazebo world with collada mesh from a GeoTIFF raster file. The colormap overlay on the mesh is customizable to any color map matplotlib supports.
 * Adding coherent noise to existing terrain
 * Cropping rasters to make them compatible with this tool chain
 * Generate synthetic terrain from coherent noise
 * Compare slice along the diagonal of .stl, .dae, and .tif files to compare rasters. Useful for debugging.
 * Deriving a simplified anomaly field from existing raster
 * Displaying rasters as figures
 * Upsample or downsample a raster

This tool is currently under active development so it is subject to change. It is provided as is. The documentation is currently incomplete and in progress.

Feel free to reach out to me with any questions you have about the project. Raising an issue  is a great way to get a hold of me.


## Under active development

A good part of the package is still under active development. These are some of the more pressing things:

 * Docstrings (about half way complete)
 * Command line interface --help (mostly done)
 * Upload package to PyPi (future thing)
 * More detailed README that explains commands (with pictures)
 * License (still contemplating this)
 * Tests (I changed the structure for packaging so paths are incorrect)

## Requirements

Other than the packages that will be automatically installed by pip, the `raster-to-gazebo-world` command needs the `blender` executable on the PATH. This is because the blender API is used to create the Collada mesh from the raster that becomes the visual and collision meshes in the Gazebo world.


## Quick start

Clone the package:

```
git clone https://github.com/franzheubach/ant-cli
```

Install the package (requires python >= 3.6):

```
python -m pip install ant-cli
```

Use the package:

```
python -m ant --help
```

## Citing

If you use this tool for your own work please cite this package in your work. Once I have an official citation available for this work I will add it. For now you can just cite this GitHub repository as a web resource.
