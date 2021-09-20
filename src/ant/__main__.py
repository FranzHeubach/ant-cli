import shutil
import json
import sys
import math
from math import floor, ceil
from pathlib import Path

import rasterio as rio
import rasterio.plot
from rasterio.enums import Resampling
from rasterio.transform import Affine
from stl.mesh import Mesh
from collada import Collada
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import click

import ant


@click.group(chain=True)
def cli():
    pass


@cli.command(name='visualize-perlin-noise', help='Visualize two-dimensional perlin noise using matplotlib')
@click.option('--size', default=1000., type=click.FLOAT, help='The target raster world size in meters. The world is square.')
@click.option('--target-resolution', default=2., type=click.FLOAT, help='The target resolution for the generated terrain.')
@click.option('--base-period', default=100., type=click.FLOAT, help='The base period of the perlin noise function used to generate the terrain.')
@click.option('--mean', default=0., type=click.FLOAT, help='The target mean of the generated terrain.')
@click.option('--amplitude', default=1., type=click.FLOAT, help='The target amplitude')
@click.option('--save-figure', type=click.Path(exists=False), help='The file path to save to figure to.')
@click.option('--octaves', default=4, type=click.INT, help='The number of perlin noise octaves combined to generate the final terrain. Successive octaves are related by lacunarity and persistence.')
@click.option('--lacunarity', default=2., type=click.FLOAT, help='The ratio of the frequency of successive perlin noise octaves. Seems counter intuitive because an octave is defined by a lacunarity of 2, but this is the vocabulary used. Usually very close to 2. Default is 2, the usual definition of an octave.')
@click.option('--persistence', default=0.5, type=click.FLOAT, help='The ratio of the amplitude of successive perlin noise octaves. Usually less than 1. Determines the roughness of the noise.')
@click.option('--seed', default=1, type=click.INT, help='The seed used for pseudo random number generation.')
def visualize_perlin_noise_command(size, target_resolution, base_period, mean, amplitude, save_figure, **noise_kwargs):
    x, y, Z = ant.create_terrain(size, target_resolution, base_period, mean, amplitude, **noise_kwargs)
    ant.terrain_plot(x, y, Z, base_period, save_figure)


@cli.command(name='augment-terrain', help='Add custom perlin noise to existing raster.')
@click.option('--target-resolution', default=2., type=click.FLOAT, help='The target resolution for the generated terrain.')
@click.option('--amplitude', default=1., type=click.FLOAT, help='The target amplitude')
@click.option('--save-figure', type=click.Path(exists=False), help='The file path to save to figure to.')
@click.option('--show-figure', default=False, type=bool, help='Whether or not to show the figure.')
@click.argument('input_raster_path', type=click.Path(exists=True))
@click.argument('output_raster_path', type=click.Path(exists=False))
@click.option('--smallest-feature', default=10., help='The desired smallest feature within the augmented terrain. Features will be at least this small. This is used with lacunarity to compute the number of octaves.')
@click.option('--lacunarity', default=2., type=click.FLOAT, help='The ratio of the frequency of successive perlin noise octaves. Seems counter intuitive because an octave is defined by a lacunarity of 2, but this is the vocabulary used. Usually very close to 2. Default is 2, the usual definition of an octave.')
@click.option('--persistence', default=0.5, type=click.FLOAT, help='The ratio of the amplitude of successive perlin noise octaves. Usually less than 1. Determines the roughness of the noise.')
@click.option('--seed', default=1, type=click.INT, help='The seed used for pseudo random number generation.')
def augment_terrain_command(target_resolution, amplitude, save_figure, show_figure, input_raster_path, output_raster_path, smallest_feature, **noise_kwargs):
    input_raster_path = Path(input_raster_path).resolve()
    output_raster_path = Path(output_raster_path).resolve()

    with rio.open(input_raster_path) as src:
        base_period = src.res[0]
        augmented_dataset = ant.augment_raster(
            src,
            target_resolution,
            amplitude,
            smallest_feature,
            **noise_kwargs
        )
        ant.save_raster(augmented_dataset, output_raster_path)
        ant.plot_raster(augmented_dataset, save_figure, show_figure, base_period, xlabel='Easting (m)', ylabel='Northing (m)', zlabel='Depth (m)')


@cli.command(name='resample-raster', help='Upsample or downsample raster.')
@click.option('--resolution-scale-factor', default=1., help='Larger than one upsamples, smaller than one downsamples.')
@click.option('--save-figure', type=click.Path(exists=False), help='The file path to save to figure to.')
@click.option('--show-figure', default=False, type=bool, help='Whether or not to show the figure.')
@click.argument('input_raster_path', type=click.Path(exists=True))
@click.argument('output_raster_path', type=click.Path(exists=False))
def resample_raster_command(resolution_scale_factor, save_figure, show_figure, input_raster_path, output_raster_path):
    input_raster_path = Path(input_raster_path).resolve()
    output_raster_path = Path(output_raster_path).resolve()

    with rio.open(input_raster_path) as src:
        resampled_dataset = ant.resample_raster(src, resolution_scale_factor, resolution_scale_factor)

    ant.save_raster(resampled_dataset, output_raster_path)
    ant.plot_raster(resampled_dataset, save_figure, show_figure, base_period=None, grid=False)


@cli.command(name='crop-raster-to-square', help='Crop raster to the minimum dimension.')
@click.argument('input_raster_path', type=click.Path(exists=True))
@click.argument('output_raster_path', type=click.Path())
def crop_raster_to_square_command(input_raster_path, output_raster_path):
    input_raster_path = Path(input_raster_path).resolve()
    output_raster_path = Path(output_raster_path).resolve()
    with rio.open(input_raster_path) as src:
        dataset = ant.crop_raster_to_square(src)
        ant.save_raster(dataset, output_raster_path)


@cli.command(name='generate-density-for-raster', help='Generate density for a raster. Sizes and placement are pulled from the raster.')
@click.option('--target-resolution', default=2., type=click.FLOAT, help='The target resolution for the generated terrain.')
@click.option('--base-period', default=100., type=click.FLOAT, help='The base period of the perlin noise function used to generate the terrain.')
@click.option('--mean', default=0., type=click.FLOAT, help='The target mean of the generated terrain.')
@click.option('--amplitude', default=1., type=click.FLOAT, help='The target amplitude')
@click.option('--save-figure', type=click.Path(exists=False), help='The file path to save to figure to.')
@click.option('--show-figure', default=False, type=bool, help='Whether or not to show the figure.')
@click.option('--terrain-path', type=click.Path(exists=True), help='The GEOTIFF terrain file for which density is being generated.')
@click.argument('output_raster_path', type=click.Path(exists=False))
@click.option('--octaves', default=4, type=click.INT, help='The number of perlin noise octaves combined to generate the final terrain. Successive octaves are related by lacunarity and persistence.')
@click.option('--lacunarity', default=2., type=click.FLOAT, help='The ratio of the frequency of successive perlin noise octaves. Seems counter intuitive because an octave is defined by a lacunarity of 2, but this is the vocabulary used. Usually very close to 2. Default is 2, the usual definition of an octave.')
@click.option('--persistence', default=0.5, type=click.FLOAT, help='The ratio of the amplitude of successive perlin noise octaves. Usually less than 1. Determines the roughness of the noise.')
@click.option('--seed', default=1, type=click.INT, help='The seed used for pseudo random number generation.')
def generate_density_command(target_resolution, base_period, mean, amplitude, save_figure, show_figure, terrain_path, output_raster_path, **noise_kwargs):
    terrain_path = Path(terrain_path).resolve()
    output_raster_path = Path(output_raster_path).resolve()

    with rio.open(terrain_path) as src:
        print(f'Original raster size: ({src.width}, {src.height})')
        # Resample the raster
        original_resolution = abs(src.res[0])

        # Determine terrain generation parameters
        size = original_resolution * src.width

        # Generate terrain
        xgen, ygen, Zgen = ant.create_terrain(size, target_resolution, base_period, mean, amplitude, **noise_kwargs)
        x = xgen*(src.res[1]/src.res[0]) + src.transform[2]
        y = ygen + src.transform[5]
        Z = Zgen

        ant.plot_raster(src, save_figure, show_figure, base_period, xlabel='Easting (m)', ylabel='Northing (m)', zlabel='Density (kg/m^3)', grid=False)
        dataset = ant.create_raster(x, y, Z)
        ant.save_raster(dataset, output_raster_path)


@cli.command(name='generate-terrain-as-raster', help='Generate terrain as raster in geotiff format')
@click.option('--world-size', default=1000., type=click.FLOAT, help='The target raster world size in meters. The world is square.')
@click.option('--target-resolution', default=2., type=click.FLOAT, help='The target resolution for the generated terrain.')
@click.option('--base-period', default=100., type=click.FLOAT, help='The base period of the perlin noise function used to generate the terrain.')
@click.option('--mean', default=0., type=click.FLOAT, help='The target mean of the generated terrain.')
@click.option('--amplitude', default=1., type=click.FLOAT, help='The target amplitude')
@click.option('--octaves', default=4, type=click.INT, help='The number of perlin noise octaves combined to generate the final terrain. Successive octaves are related by lacunarity and persistence.')
@click.option('--lacunarity', default=2., type=click.FLOAT, help='The ratio of the frequency of successive perlin noise octaves. Seems counter intuitive because an octave is defined by a lacunarity of 2, but this is the vocabulary used. Usually very close to 2. Default is 2, the usual definition of an octave.')
@click.option('--persistence', default=0.5, type=click.FLOAT, help='The ratio of the amplitude of successive perlin noise octaves. Usually less than 1. Determines the roughness of the noise.')
@click.option('--seed', default=1, type=click.INT, help='The seed used for pseudo random number generation.')
@click.option('--record-parameters', default=True, type=click.BOOL, help='Whether to record the parameters used in a file in the same directory as the generated terrain.')
@click.argument('file-path')
def generate_terrain_command(**kwargs):
    ant.generate_terrain(**kwargs)


@cli.command(name='derive-anomaly-field', help='Derive the gravity anomaly field from bathymetry, with the optional inclusion of density.')
@click.option('--interpolation-depth', default=-20., help='The height at which the gravity anomaly is calculated.')
@click.option('--target-window-size', prompt=True, type=click.FLOAT, help='The size of the window used to sweep the terrain to calculate the anomaly.')
@click.option('--target-resolution', prompt=True, type=click.FLOAT, help='The target resolution of the derived anomaly map.')
@click.option('--record-parameters', default=True, type=click.BOOL, help='Record the input parameters in a json file.')
@click.option('--density-file-path', type=click.Path(exists=True), help='Optional inclusion of density field in the calculation.')
@click.argument('bathymetry_file_path', type=click.Path(exists=True))
@click.argument('anomaly_file_path', type=click.Path())
def derive_anomaly_field_command(**kwargs):
    ant.derive_anomaly_field(**kwargs)


@cli.command(name='show-tiff', help='Show and optionally save a figure of a geotif')
@click.option('--save-figure', type=click.Path(exists=False), help='The file path to save to figure to.')
@click.option('--x-label', default='X (m)', help='The x-axis label for the figure')
@click.option('--y-label', default='Y (m)', help='The y-axis label for the figure')
@click.option('--z-label', default='Depth (m)', help='The colormap label for the figure')
@click.option('--x-tick-spacing', type=float, help='The spacing used for the x ticks.')
@click.option('--y-tick-spacing', type=float, help='The spacing used for the y ticks.')
@click.option('--x-scale-factor', default=1, type=float, help='The scale factor to apply to the x values before plotting. Useful for unit conversions.')
@click.option('--y-scale-factor', default=1, type=float, help='The scale factor to apply to the y values before plotting. Useful for unit conversions.')
@click.argument('input_geotiff_raster', type=click.Path(exists=True))
def show_tiff_command(save_figure, input_geotiff_raster, x_label, y_label, z_label, x_tick_spacing, y_tick_spacing, x_scale_factor, y_scale_factor):
    input_geotiff_raster = Path(input_geotiff_raster).resolve()
    show_figure = True
    with rio.open(input_geotiff_raster) as src:
        base_period = src.res[0]
        ant.plot_raster(
            src,
            save_figure,
            show_figure,
            base_period,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
        )


@cli.command(name='translate-raster', help='Translate a raster by a certain amount in the x and y')
@click.argument('input_raster', type=click.Path(exists=True))
@click.argument('dx', type=float)
@click.argument('dy', type=float)
@click.argument('output_raster', type=click.Path())
def translate_raster_command(input_raster, dx, dy, output_raster):
    input_raster = Path(input_raster).resolve()
    output_raster = Path(output_raster).resolve()
    with rio.open(input_raster) as src:
        profile = src.profile
        profile.update(
            transform=rasterio.Affine(
                src.transform[0],
                src.transform[1],
                src.transform[2]+dx,
                src.transform[3],
                src.transform[4],
                src.transform[5]+dy
            )
        )
        with rio.open(output_raster,'w', **profile) as dst:
            dst.write(src.read(1),1)


@cli.command(name='compare-slices', help='Supports slice of .tif rasters and .stl meshes. Is useful to compare for acuracy of the mesh.')
@click.option('--n', default=100, help='Number of points along the slice.')
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def compare_slices_command(n, files):
    for file in files:
        path = Path(file).resolve()
        print(f'Processing {path.name}')
        if path.suffix == '.stl':
            m = Mesh.from_file(file)
            distances, z = ant.slice_stl(m, n=n)
        elif path.suffix == '.tif':
            with rio.open(file) as src:
                distances, z = ant.slice_raster(src, n=n)
        elif path.suffix == '.dae':
            m = Collada(file)
            distances, z = ant.slice_collada(m, n=n)
        else:
            print(f'Skipping {file} because extension not supported.')
            continue
        plt.plot(distances, z, label=path.name, lw=1)
    plt.legend()
    plt.show()


@cli.command(name='raster-to-gazebo-world', help='Convert geotiff raster into gazebo world')
@click.option('--world-size', default=1000., prompt='Gazebo world size in meters', help='The target world size within gazebo in meters. The world will be square.')
@click.option('--ros-gazebo-package-folder', type=click.Path(exists=True), help='Path to the directory used for containing the output gazebo world files.')
@click.option('--target-mesh-resolution', default=5., help='The target mesh resolution for the collision and visual meshes used for the gazebo world.')
@click.option('--height-exaggeration', default=1., help='The exaggeration of the height axis for the collision and visual meshes within the gazebo world.')
@click.option('--overwrite', type=click.BOOL, default=False, help='Overwrite already existing world of the same name.')
@click.option('--colormap', default=None, help='Matplotlib colormap identifier for the mesh texture.')
@click.argument('input-geotiff-raster', type=click.Path(exists=True))
@click.argument('gazebo-world-name')
def geotiff2gazebo(world_size, ros_gazebo_package_folder, target_mesh_resolution, height_exaggeration, overwrite, colormap, input_geotiff_raster, gazebo_world_name):
    source_tiff = Path(input_geotiff_raster).resolve()
    ros_gazebo_package_folder = Path(ros_gazebo_package_folder).resolve()
    name = gazebo_world_name
    target_size = world_size

    print('Deriving paths..')
    source_tiff=source_tiff
    ros_gazebo_package_folder=ros_gazebo_package_folder
    worlds_folder=ros_gazebo_package_folder / 'worlds'
    models_folder=ros_gazebo_package_folder / 'models'
    path_model_directory = models_folder / name
    textures_folder=ros_gazebo_package_folder / 'media' / 'materials' / 'textures'
    path_tiff=path_model_directory / (source_tiff.stem + '_downsampled.tif')
    path_stl=path_model_directory / (path_tiff.stem + '.stl')
    path_collada=path_model_directory / (path_tiff.stem + '_textured.dae')
    path_sdf=path_model_directory / 'model.sdf'
    path_config=path_model_directory / 'model.config'
    path_world=worlds_folder / f'{name}.world'
    path_open_world_script=Path(f'./open_{name}_in_gazebo.sh')
    path_texture=textures_folder / f'{name}_colormap.png'
    path_heightmap_image = path_model_directory / f'{name}_heightmap.png'

    if path_model_directory.exists() and overwrite:
        print('Removing previous version of this world..')
        shutil.rmtree(path_model_directory)
    if path_model_directory.exists() and not overwrite:
        print('Model directory already exists. Use --overwrite to enable overwriting previous worlds.')
        sys.exit(1)
    print('Create model directory..')
    path_model_directory.mkdir(parents=True, exist_ok=True)
    print('Copying over source geotiff raster..')
    shutil.copyfile(source_tiff, path_model_directory / source_tiff.name)

    print(f'Downsampling raster to {target_mesh_resolution}m resolution for collision and visual mesh..')

    with rio.open(source_tiff) as src:
        print(src.bounds.top - src.bounds.bottom)
        x_res_factor = abs(target_mesh_resolution / src.res[0])
        y_res_factor = abs(target_mesh_resolution / src.res[1])
        print(f'Derived resolution factor to be applied. ({x_res_factor}, {y_res_factor})')
        resampled_raster = ant.resample_raster(src, x_res_factor, y_res_factor)
        ant.save_raster(resampled_raster, path_tiff)
        raster_info = ant.raster_info(resampled_raster)
        print(raster_info)


    print('Creating color map texture PNG..')
    ant.create_colormap_texture(source_tiff, path_texture, colormap)

    print('Creating image texture for blender..')
    blender_parameters = ant.create_image_texture_for_blender(path_tiff, path_heightmap_image)

    print('Creating solidified and textured collada mesh..')
    scale = 1 / ((raster_info.xmax - raster_info.xmin) / target_size)
    height_extent = abs(raster_info.zmax - raster_info.zmin)
    ant.create_textured_collada_file(path_heightmap_image, path_collada, path_texture, scale, target_size, target_mesh_resolution, height_exaggeration, height_extent, blender_parameters)
    print('Writing out model files..')
    ant.create_model_files(name, path_config, path_sdf, scale, height_exaggeration, raster_info, path_collada, models_folder, overwrite=overwrite)
    print('Writing out world file..')
    ant.create_world_file(name, path_world, overwrite=overwrite)
    print('Creating convenience bash script to launch world within gazebo..')
    ant.create_open_gazebo_world_script(name, path_open_world_script, ros_gazebo_package_folder, worlds_folder, models_folder, overwrite=overwrite)
    print('Done')
    print(f'World can be launched with "bash {path_open_world_script}"')


if __name__ == '__main__':
    cli()

