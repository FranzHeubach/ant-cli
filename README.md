# AUV Navigation Testbed

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

## Under active development

A good part of the package is still under active development. These are some of the more pressing things:

 * Docstrings (about half way complete)
 * Command line interface --help (mostly done)
 * Upload package to PyPi (future thing)
 * More detailed README that explains commands (with pictures)
 * License (still contemplating this)
 * Tests (I changed the structure for packaging so paths are incorrect)

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