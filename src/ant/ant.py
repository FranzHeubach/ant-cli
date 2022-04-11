import tempfile
import os
import json
import math
from math import floor, ceil, log, sqrt
import random
import shlex
import sys
import stat
import textwrap
import subprocess
from collections import namedtuple
from pathlib import Path
import functools

import numpy as np
import scipy.interpolate
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from perlin_noise import PerlinNoise
from stl import mesh
import rasterio as rio
import rasterio.transform
import rasterio.plot
from rasterio.enums import Resampling
from rasterio import Affine, MemoryFile
from rasterio.crs import CRS
import rasterio.fill


def crop_raster_to_square(src):
    """
    Crops a raster to have the same width and height.

    The cropping process uses the minimum dimension of the height and width.

    :param      src:  The original rasterio dataset
    :type       rasterio dataset

    :returns:   The cropped rasterio dataset.
    :rtype:     rasterio dataset
    """
    data = src.read(1)
    min_shape = min(*data.shape)
    data = data[:min_shape, :min_shape]
    dst = MemoryFile().open(
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        dtype=data.dtype,
        transform=src.transform,
        count=1
    )
    dst.write(data, 1)
    return dst


def extract_arrays_from_raster(src):
    """
    Extract the x, and y vectors as well as the z matrix from a rasterio data
    set.

    The x vector will contain the x coordinates in ascending column index order.
    This means the x coordinate values are not guaranteed to be in ascending
    order. The y vector will contain the y coordinates in ascending row order.
    This means the y coordinate values are not guaranteed to the in ascending
    order. The z matrix will contain the rasterio data set values. The x and y
    vectors are return this way so you can index using a (row, col) index. The
    value of z at (row, col) corresponds to the x[col] x coordinate and the
    y[row] y coordinate.

    :param      src:             The source rasterio data set.
    :type       src:             rasterio dataset

    :returns:   A tuple of the x coordinate vector, the y coordinate vector, and
                the z value matrix.
    :rtype:     tuple (1 x n np.array, 1 x m np.array, m x n np.array)
    """
    ivec = range(src.height)
    ivec_const = [0 for j in range(src.width)]
    jvec = range(src.width)
    jvec_const = [0 for i in range(src.height)]
    assert len(ivec) == len(jvec_const)
    assert len(jvec) == len(ivec_const)
    x_jvec, _ = rasterio.transform.xy(src.transform, ivec_const, jvec)
    _, y_ivec = rasterio.transform.xy(src.transform, ivec, jvec_const)
    x_jvec = np.array(x_jvec)
    y_ivec = np.array(y_ivec)
    return x_jvec, y_ivec, src.read(1)


def resample_raster(src, x_resolution_factor, y_resolution_factor, resampling=None):
    """
    Resample a raster at a higher or lower resolution.

    The returned raster height is given by int(src.height / y_resolution_factor)
    and the returned raster width is given by int(src.width /
    x_resolution_factor). Therefore, the output raster is not guaranteed to
    satisfy the resolution factor exactly.

    Warning: Resampling does not handle no data areas correctly.

    :param      src:                  The source rasterio data set.
    :type       src:                  rasterio dataset
    :param      x_resolution_factor:  The factor the x resolution is multiplied
                                      by. A number less than one will
                                      interpolate to get a higher resolution. A
                                      number less than one will sub-sample to
                                      get a lower resolution. Must be greater
                                      than 0.
    :type       x_resolution_factor:  float
    :param      y_resolution_factor:  The factor the y resolution is multiplied
                                      by. A number less than one will
                                      interpolate to get a higher resolution. A
                                      number less than one will sub-sample to
                                      get a lower resolution. Must be greater
                                      than 0.
    :type       y_resolution_factor:  float
    :param      resampling:           The method used for resampling the source
                                      data set to produce the output data set
                                      with the desired resolution.
    :type       resampling:           str

    :returns:   The resampled rasterio data set.
    :rtype:     rasterio dataset
    """
    data = src.read(
        out_shape=(
            src.count,
            int(src.height / y_resolution_factor),
            int(src.width / x_resolution_factor)
        ),
        resampling=resampling or Resampling.bilinear
    )

    data = data[0,:,:]

    # scale image transform
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

    dst = MemoryFile().open(
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform,
    )
    dst.write(data,1)
    return dst


def transform_raster_to_origin(src, size):
    """
    Transform raster to origin and scale to a specific size.

    The output raster will be square, will be scaled to size, and have the
    bottom left corner at the origin. If there are no data values, they will be
    maintained in the output dataset by transfering the mask information.

    :param      src:   The source raster. Must be square.
    :type       src:   rasterio dataset
    :param      size:  The size of the ouput raster.
    :type       size:  float

    :returns:   The output raster
    :rtype:     rasterio datasets
    """
    data = src.read(1)
    x_pixel_size = size / data.shape[1]
    y_pixel_size = size / data.shape[0]

    transform = rio.transform.from_origin(
        west=0,
        north=size,
        xsize=x_pixel_size,
        ysize=y_pixel_size)

    dst = MemoryFile().open(
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform)

    dst.write(data,1)
    dst.write_mask(src.read_masks(1))
    return dst


def create_raster(x_jvec, y_ivec, Z, crs=None):
    """
    Creates a rasterio data set using the x and y coordinate vectors, and the Z
    value matrix.

    :param      x_jvec:          The x coordinate vector. The x coordinate
                                 vector must be the same length as the value
                                 matrix has columns. The x coordinate vector
                                 must also be in column index ascending order.
    :type       x_jvec:          np.array
    :param      y_ivec:          The y coordinate vector. The y coordinate
                                 vector  must be the same length as the value
                                 matrix has rows. The y coordinate vector must
                                 also in row index ascending order.
    :type       y_ivec:          np.array
    :param      Z:               The z value matrix. The z matrix contains the
                                 source values. The value of z at (row, col)
                                 must correspond to the x[col] x coordinate and
                                 the y[row] y coordinate.
    :type       Z:               np.array
    :param      crs:             The coordinate reference system that these
                                 coordinates use. Defaults to EPSG code 26920.
                                 You can look up the corresponding zone in the
                                 EPSG database online.
    :type       crs:             rasterio.CRS

    :returns:   The output rasterio data set.
    :rtype:     rasterio dataset

    :raises     AssertionError:  Raised if the size of the vectors are not
                                 greater than 2 or if the vector size does not
                                 match the value matrix size.
    """
    assert len(x_jvec) >= 2
    assert len(y_ivec) >= 2
    assert len(y_ivec) == Z.shape[0]
    assert len(x_jvec) == Z.shape[1]
    x_res = (x_jvec[1] - x_jvec[0])
    y_res = (y_ivec[1] - y_ivec[0])
    transform = Affine.translation(x_jvec[0] - x_res / 2, y_ivec[-1] + y_res / 2) * Affine.scale(x_res, -y_res)
    dst = MemoryFile().open(
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype,
        crs=crs or CRS.from_epsg(26920),
        transform=transform,
    )
    dst.write(Z,1)
    return dst


def save_raster(src, path):
    """
    Save a rasterio data set to a file.

    :param      src:   The source rasterio data set.
    :type       src:   rasterio dataset
    :param      path:  The path to the output file.
    :type       path:  Pathlike

    :returns:   The resolved path to the output file.
    :rtype:     pathlib.Path
    """
    with rio.open(path, 'w', **src.profile) as dst:
        dst.write(src.read())
    return Path(path).resolve()


def prepare_interpolation_function(src):
    """
    Prepare an interpolation function using a rasterio data set.

    The function can be called with an array of points and they will be
    interpolated using scipy's regular grid interpolatio function.

    :param      src:  The source rasterio data set that will be interpolated.
    :type       src:  rasterio dataset

    :returns:   A function with the signature (nx2 np.array of points) that
                returns a nx1 array of interpolated values.
    :rtype:     func
    """
    x_jvec, y_ivec, Z = extract_arrays_from_raster(src)

    # Manipulate dimensions to satisfy the strictly ascending rule for
    # scipy.interpolate.interpn
    x_ascending = x_jvec[1] > x_jvec[0]
    y_ascending = y_ivec[1] > y_ivec[0]

    if not x_ascending:
        x_jvec = x_jvec[::-1]
        Z = np.flip(Z, axis=1)
        print('Flipped x vector, and axis 1 on Z (columns)')

    if not y_ascending:
        y_ivec = y_ivec[::-1]
        Z = np.flip(Z, axis=0)
        print('Flipped y vector, and axis 0 on Z (rows)')

    points = (x_jvec, y_ivec)
    return functools.partial(scipy.interpolate.interpn, points, np.transpose(Z))


### Below have not been thoroughly tested

def octaves_from_smallest_feature(size, base_period, lacunarity):
    """
    Number of perlin noise octaves to use base on smallest desired feature.

    :param      size:         The size of the smallest desired feature.
    :type       size:         float
    :param      base_period:  The base Perlin noise function's period. Is the
                              largest period or lowest frequency in the Perlin
                              noise sequence.
    :type       base_period:  float
    :param      lacunarity:   The ratio between successive perlin noise function
                              frequencies. Usually 2.
    :type       lacunarity:   float

    :returns:   The number of octaves or successive perlin noise functions
                needed to get at smallest feature size of at least the desired
                size.
    :rtype:     int
    """
    # Generates features at least that small
    octaves = math.ceil((log(size) - log(base_period)) / log(1/lacunarity) + 1)
    return octaves


def augment_raster(src, target_resolution, amplitude, smallest_feature, lacunarity, persistence, seed):
    """
    Augment an existing raster data set using Perlin noise.

    For the original source raster values to be preserved the lacunarity must be
    a multiple of 2. This function only supports square rasters and will throw
    an exception otherwise.

    :param      src:                The source rasterio data set. The data set
                                    must be square.
    :type       src:                rasterio dataset
    :param      target_resolution:  The target resolution of the output rasterio
                                    data set. This resolution is not guaranteed
                                    to be met, but it will be close.
    :type       target_resolution:  float
    :param      amplitude:          The amplitude of the largest perlin noise
                                    terrain variation. This is measured as the
                                    total amount of Perlin noise variation.
    :type       amplitude:          float
    :param      smallest_feature:   The smallest feature that is guaranteed to
                                    exist in the augmented terrain. The smallest
                                    period of the Perlin noise function with the
                                    highest frequency will at least this small.
    :type       smallest_feature:   float
    :param      lacunarity:         The ratio of successive Perlin noise
                                    function's frequencies. Must be a multiple
                                    of 2 to maintain the integrity of source
                                    data sets measurements.
    :type       lacunarity:         float
    :param      persistence:        The ratio of successive Perlin noise
                                    function's amplitudes. Controls visual
                                    roughness.
    :type       persistence:        float.
    :param      seed:               The seed for the pseudo-random number
                                    generator used for the Perlin noise.
    :type       seed:               int

    :returns:   Augmented raster data set.
    :rtype:     rasterio dataset

    :raises     AssertionError:     Raised when the raster is not square.
    """
    assert src.width == src.height or not 'Only square rasters supported'

    # Resample the raster
    original_resolution = src.res[0]
    print(f'Original resolution: {original_resolution}')
    resolution_factor = target_resolution / original_resolution
    resampled_dataset = resample_raster(src, resolution_factor, resolution_factor)
    print(f'Original start {rio.transform.xy(src.transform, 0,0)}, end {rio.transform.xy(src.transform, src.height, src.width)}')

    # Determine terrain generation parameters
    new_res = resampled_dataset.res[0]
    size = abs(
        rio.transform.xy(src.transform, 0, 0)[0]
        - rio.transform.xy(src.transform, src.height-1, src.width-1)[0])
    print(size)
    base_period = original_resolution
    mean = 0
    octaves = octaves_from_smallest_feature(smallest_feature, base_period, lacunarity)
    if octaves < 1:
        octaves = 1

    # Generate terrain
    xgen, ygen, Zgen = create_terrain(
        size,
        target_resolution,
        base_period,
        mean,
        amplitude,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
        seed=seed
    )

    # Augment the existing terrain with the generated terrain
    dx, dy = rio.transform.xy(src.transform, 0, 0)
    Z = resampled_dataset.read(1).astype(np.float64)

    # Place the augmented matrix in the middle of the resampled raster
    di = int(round(abs(Zgen.shape[0] - Z.shape[0]) / 2))
    dj = int(round(abs(Zgen.shape[0] - Z.shape[0]) / 2))
    Z[di:di+Zgen.shape[0],dj:dj+Zgen.shape[1]] += Zgen
    x_jvec_resampled, y_ivec_resampled, _ = extract_arrays_from_raster(resampled_dataset)
    x_jvec_resampled, y_ivec_resampled = np.array(x_jvec_resampled), np.array(y_ivec_resampled)
    x = x_jvec_resampled
    x[dj:dj+Zgen.shape[0]] += (xgen + dx)
    y = y_ivec_resampled
    y[di:di+Zgen.shape[1]] += (ygen + dy)

    return create_raster(x, y, Z, crs=resampled_dataset.crs)


def round_down_to_odd(num):
    """
    Round a number down to the nearest odd number.

    :param      num:  The number
    :type       num:  float

    :returns:   The nearest odd number lower than the number.
    :rtype:     int
    """
    return int(math.ceil(num) // 2 * 2 - 1)


def local_anomaly(bathy, bathy_xj, bathy_yi, interpolation_depth, i, j, m, n, density):
    """
    Calculate the local gravity anomaly using a window on a data set.

    This calculated anomaly is not any official measure of the gravity anomaly.
    It is a simplified version that approximates the gravity anomaly gradients
    but is not correct in the sense of absolute values.

    This function is still under active development.

    The function assumes a few parameter values. An average crustal density of
    2670 kg/m^3, an average seawater density of 1027 kg/m^3, a universal gravity
    constant of 6.674e-11 m^3 kg^-1 s^-2.

    :param      bathy:                The bathymetry value matrix. The value at
                                      row, col must correspond with the
                                      coordinate (x[col], y[row]).
    :type       bathy:                d x p np.array
    :param      bathy_xj:             The x coordinate vector. Length must be
                                      the same as the columns in bathy value
                                      matrix.
    :type       bathy_xj:             1 x p np.array
    :param      bathy_yi:             The y coordinate vector. Length must be
                                      the same as the rows in bathy value
                                      matrix.
    :type       bathy_yi:             1 x d np.array
    :param      interpolation_depth:  The height at which the gravity anomaly
                                      should be calculated. Positive heights are
                                      above the water surface. Negative heights
                                      are below the water surface.
    :type       interpolation_depth:  float
    :param      i:                    The starting row of the window.
    :type       i:                    int
    :param      j:                    The starting column of the window.
    :type       j:                    int
    :param      m:                    The number of rows in the window.
    :type       m:                    int
    :param      n:                    The number of columns in the window.
    :type       n:                    int
    :param      density:              The density value matrix. The value at
                                      row, col must correspond with the
                                      coordinate (x[col], y[row]). The average
                                      crustal density is used if no density
                                      matrix is provided.
    :type       density:              d x p np.array

    :returns:   A tuple of length 3. (x coordinate, y coordinate, gravity
                anomaly)
    :rtype:     tuple (float, float, float)
    """

    crustal_density = 2670  # kg/m^3
    # The density of sea water.
    seawater_density = 1027  # kg/m^3
    # The universal gravitation constant.
    G = 6.674e-11 # m^3 kg^-1 s^-2
    area = np.abs((bathy_xj[1] - bathy_xj[0]) * (bathy_yi[1]-bathy_yi[0]))

    # m and n must be odd
    # i, j, m, n are all in terms of indices.
    # the location that the anomaly is being calculated at.
    min_res = max(abs(bathy_xj[1]-bathy_xj[0]), abs(bathy_yi[1]-bathy_yi[0]))
    x, y = bathy_xj[j+(floor(n/2))], bathy_yi[i+(floor(m/2))]
    xx, yy = np.meshgrid(bathy_xj[j:j+n], bathy_yi[i:i+m])
    dxx = np.abs(xx - x)
    dyy = np.abs(yy - y)
    height = bathy[i:i+m, j:j+n]
    dzz = interpolation_depth - height / 2.
    if density is None:
        density = crustal_density * np.ones(dxx.shape)

    r = np.sqrt(np.square(dxx) + np.square(dyy) + np.square(dzz))
    effective_density = np.zeros(dxx.shape)
    np.putmask(effective_density, height>=0, density)
    np.putmask(effective_density, height<0, seawater_density-density)

    # Remove calculations using terrain that is too close
    too_close_mask = r < min_res / 2.
    r[too_close_mask] = np.nan
    if (np.sum(too_close_mask) > 1):
        print('WARNING: Threw out {} cells because they were to close.')
        print(np.sum(too_close_mask))

    volume = area * np.abs(height)
    dzz = np.abs(dzz)
    effective_mass = effective_density * volume
    gravity_contribution = G * effective_mass * dzz / r**3
    gravity_contribution[np.isnan(gravity_contribution)] = 0
    return x, y, np.sum(gravity_contribution)


def local_anomaly_sweep(bathy, bathy_xj, bathy_yi, interpolation_depth, m, n, istep, jstep, density):
    """
    Perform a sweep across a larger matrix to calculate multiple local gravity
    anomalies using a moving window.

    :param      bathy:                The bathymetry value matrix. The value at
                                      row, col must correspond with the
                                      coordinate (x[col], y[row]).
    :type       bathy:                d x p np.array
    :param      bathy_xj:             The x coordinate vector. Length must be
                                      the same as the columns in bathy value
                                      matrix.
    :type       bathy_xj:             1 x p np.array
    :param      bathy_yi:             The y coordinate vector. Length must be
                                      the same as the rows in bathy value
                                      matrix.
    :type       bathy_yi:             1 x d np.array
    :param      interpolation_depth:  The height at which the gravity anomaly
                                      should be calculated. Positive heights are
                                      above the water surface. Negative heights
                                      are below the water surface.
    :type       interpolation_depth:  float
    :param      m:                    The number of rows in the window.
    :type       m:                    int
    :param      n:                    The number of columns in the window.
    :type       n:                    int
    :param      istep:                The number of rows to skip for every
                                      window step.
    :type       istep:                int
    :param      jstep:                The number of columns to skip for every
                                      window step.
    :type       jstep:                int
    :param      density:              The density value matrix. The value at
                                      row, col must correspond with the
                                      coordinate (x[col], y[row]). The average
                                      crustal density is used if no density
                                      matrix is provided.
    :type       density:              d x p np.array

    :returns:   A tuple of length 3. (x coordinate vector, y coordinate vector,
                anomaly value matrix)
    :rtype:     tuple (1 x q np.array, 1 x r np.array, r x q np.array)
    """
    ivec = range(0, bathy.shape[0] - m, istep)
    jvec = range(0, bathy.shape[1] - n, jstep)

    # Populate the anomaly field
    anomaly = np.zeros((len(ivec), len(jvec)))
    xs = list()
    ys = list()

    # Calculate the local anomaly
    for i, ibathy in tqdm(enumerate(ivec), desc='Calculating anomalies', total=len(ivec), unit='anomaly'):
        for j, jbathy in enumerate(jvec):
            x, y, anomaly[i,j] = local_anomaly(bathy, bathy_xj, bathy_yi, interpolation_depth, ibathy, jbathy, m, n, density)
            if (i == 0):
                xs.append(x)
        ys.append(y)

    return np.asarray(xs), np.asarray(ys), anomaly


def match_density_to_bathymetry(density_src, bathy, bathy_xj, bathy_yi):
    """
    Match the density field shape to the bathymetry field shape.

    Useful for when the density array is of lower resolution and needs to be
    interpolated to match the shape of the bathymetry value matrix.

    :param      density_src:  The crustal density source raster.
    :type       density_src:  rasterio dataset
    :param      bathy:        The bathymetry value matrix. The value at row, col
                              must correspond with the coordinate (x[col],
                              y[row]).
    :type       bathy:        d x p np.array
    :param      bathy_xj:     The x coordinate vector. Length must be the same
                              as the columns in bathy value matrix.
    :type       bathy_xj:     1 x p np.array
    :param      bathy_yi:     The y coordinate vector. Length must be the same
                              as the rows in bathy value matrix.
    :type       bathy_yi:     1 x d np.array

    :returns:   The density value matrix with the same shape as the bathymetry
                value matrix.
    :rtype:     d x p np.array
    """
    interpolate_density = prepare_interpolation_function(density_src)

    density = np.zeros(bathy.shape)
    for j in range(density.shape[1]):
        x_interp = (bathy_xj[j] * np.ones(bathy_xj.size)).reshape(-1,1)
        y_interp = bathy_yi.reshape(-1,1)
        interp_points = np.concatenate((x_interp, y_interp), axis=1)
        density[:,j] = interpolate_density(interp_points)
    return density


def derive_anomaly_field(bathymetry_file_path, anomaly_file_path, density_file_path, interpolation_depth, target_window_size, target_resolution, record_parameters=True):
    """
    Derive the gravity anomaly file directly from and to files in the
    filesystem. For more details on how the gravity anomaly is calculated see
    local_anomaly.

    :param      bathymetry_file_path:  The source bathymetry file path.
    :type       bathymetry_file_path:  Pathlike
    :param      anomaly_file_path:     The output gravity anomaly file path.
    :type       anomaly_file_path:     Pathlike
    :param      density_file_path:     The source crustal density file path. If
                                       not provided a constant density will be
                                       assumed.
    :type       density_file_path:     Pathlike
    :param      interpolation_depth:   The height at which the gravity anomaly
                                       should be calculated. Positive heights
                                       are above the water surface. Negative
                                       heights are below the water surface.
    :type       interpolation_depth:   float
    :param      target_window_size:    The target window size used for the local
                                       gravity anomaly sweep. In meters. Is not
                                       guaranteed to be exactly satisfied.
    :type       target_window_size:    float
    :param      target_resolution:     The target resolution of the gravity
                                       anomaly resolution. Controls the window
                                       sweep step size. Not guaranteed to be
                                       exactly satisfied.
    :type       target_resolution:     float
    :param      record_parameters:     Output a file that records the parameters
                                       used as input to this function in same
                                       place as the anomaly_file_path.
    :type       record_parameters:     bool
    """
    bathy_file_path = Path(bathymetry_file_path).resolve()
    anomaly_file_path = Path(anomaly_file_path).resolve()
    outfile = Path(anomaly_file_path).resolve()

    print('Reading input..')
    with rio.open(bathy_file_path, 'r') as src:
        _, bathy_yi = rio.transform.xy(src.transform, range(src.shape[0]), np.zeros(src.shape[1]))
        bathy_xj, _ = rio.transform.xy(src.transform, np.zeros(src.shape[0]), range(src.shape[1]))
        bathy_yi, bathy_xj = np.array(bathy_yi), np.array(bathy_xj)
        bathy = src.read(1)

    if density_file_path:
        density_file_path = Path(density_file_path).resolve()
        with rio.open(density_file_path) as src:
            _, density_yi = rio.transform.xy(src.transform, range(src.shape[0]), np.zeros(src.shape[1]))
            density_xj, _ = rio.transform.xy(src.transform, np.zeros(src.shape[0]), range(src.shape[1]))
            density_yi, density_xj = np.array(density_yi), np.array(density_xj)
            density = src.read(1)
            if density.shape != bathy.shape:
                density = match_density_to_bathymetry(src, bathy, bathy_xj, bathy_yi)
    else:
        density = None

    # calculate the window size (index) based on the target window size
    # (meters). The window size must be rounded to the nearest odd integer
    # to work with the local anomaly function.
    x_res = abs(bathy_xj[1] - bathy_xj[0])
    y_res = abs(bathy_yi[1] - bathy_yi[0])
    m = round_down_to_odd(target_window_size / y_res)
    n = round_down_to_odd(target_window_size / x_res)
    istep = int(round(target_resolution / y_res))
    jstep = int(round(target_resolution / x_res))

    print(f'Bathy shape: {bathy.shape}')
    print(f'Bathy x resolution: {x_res:.2f} m')
    print(f'Bathy y resolution: {y_res:.2f} m')

    parameters = {
        'bathymetry_file_path': bathymetry_file_path,
        'anomaly_file_path': str(anomaly_file_path),
        'density_file_path': str(density_file_path),
        'interpolation_depth': interpolation_depth,
        'target_window_size': target_window_size,
        'target_resolution': target_resolution,
    }
    if record_parameters:
        parameters_path = anomaly_file_path.stem + '_parameters.json'
        with open(parameters_path, 'w') as outfile:
            json.dump(parameters, outfile)

    print('Calculating anomaly..')
    print(f'Window size (x by y): {x_res*n:.2f} m by {y_res*m:.2f} m')
    print(f'Window step (x, y): {x_res*jstep:.2f} m, {y_res*istep:.2f} m')
    print(f'Window size (i x j): {m} x {n}')
    print(f'Window step (i, j): {istep}, {jstep}')
    x, y, anomaly = local_anomaly_sweep(bathy, bathy_xj, bathy_yi, interpolation_depth, m, n, istep, jstep, density)

    print('Writing out anomaly file..')
    dst = create_raster(x, y, anomaly)
    save_raster(dst, anomaly_file_path)
    with rio.open(anomaly_file_path, 'r') as src:
        print(f'Resolution: {src.res}')
        print(f'Bounds: {src.bounds}')


def noise(fraction, periods=1, octaves=4, lacunarity=2, persistence=0.5, seed=1):
    """
    Summation of lacunarity.

    :param      fraction:     The iterable of fractions from 0 to 1 passed into
                              the PerlinNoise function. This number of values
                              determines the dimension of the perlin noise.
    :type       fraction:     iterable with floats with domain [0,1]
    :param      periods:      The number of periods that the base frequency
                              perlin noise function has within the domain. The
                              default is 1.
    :type       periods:      int
    :param      octaves:      The number of successive perlin noise functions
                              used. Relates to detail. The lacunarity and
                              persistence parameters determine the relationships
                              between successive perlin noise functions in terms
                              of frequency and amplitude, respectively. The
                              default is 4.
    :type       octaves:      int
    :param      lacunarity:   The ratio of the frequency of successive perlin
                              noise octaves. Seems counter intuitive because an
                              octave is defined by a lacunarity of 2, but this
                              is the vocabulary used. Usually very close to 2.
                              Default is 2, the usual definition of an octave.
    :type       lacunarity:   float
    :param      persistence:  The ratio of the amplitude of successive perlin
                              noise octaves. Usually less than 1. Determines the
                              roughness of the noise.
    :type       persistence:  float with domain (0,1)
    :param      seed:         The seed used for initializing the pseudo random
                              number generator.
    :type       seed:         int

    :returns:   Evaluated noise
    :rtype:     float
    """
    random.seed(seed)
    noise = 0
    for k in range(octaves):
        n = PerlinNoise(octaves=periods*lacunarity**k, seed=seed)
        noise += persistence**k * n(fraction)
    return noise


def create_terrain(size, resolution, base_period, mean, amplitude, **noise_kwargs):
    """
    Creates synthetic terrain using a combination of Perlin noise at different
    octaves and amplitudes.

    Since in mathematics matrices use the row-major-order the origin is in the
    top left of the matrix. As such:
        0/0---column--->
         |
         |
        row
         |
         |
         v
    This is opposite of the Cartesian coordinate system where the abscissa
    (horizontal axis) is designated by the x coordinate, and the ordinate
    (vertical axis) is designated by the y coordinate. This leaves us with
    this:
        0/0---X--->
         |
         |
         Y
         |
         |
         v
    Therefore, the x value is incrementing on the inside of the loop within a
    matrix row, or in other words along the column indexed direction.

    :param      size:          The size of the generated terrain in meters.
    :type       size:          float
    :param      resolution:    The resolution of the generated terrain in
                               meters.
    :type       resolution:    float
    :param      base_period:   The period of the base perlin noise function in
                               the series perlin noise function octaves in
                               meters.
    :type       base_period:   float
    :param      mean:          The mean of the generated terrain.
    :type       mean:          float
    :param      amplitude:     The amplitude of the generated terrain. This is
                               the max value minus the min value of the
                               generated terrain.
    :type       amplitude:     float
    :param      noise_kwargs:  The keywords arguments passed to the noise
                               function.
    :type       noise_kwargs:  dictionary

    :returns:   The x vector, the y vector, and the terrain height value matrix.
    :rtype:     (n x 1 numpy.array, n x 1 numpy.array, n x n numpy.array)
    """
    xs = np.arange(0, size+1, resolution)
    ys = np.arange(0, size+1, resolution)
    x_fracs = xs / size
    y_fracs = ys / size
    periods = math.ceil(size / base_period)

    pic = []
    for y_frac in tqdm(y_fracs, 'Generating Terrain'):
        row = []
        for x_frac in x_fracs:
            row.append(noise((x_frac, y_frac), periods=periods, **noise_kwargs))
        pic.append(row)

    Z = np.asarray(pic)
    # Z = Z - Z.mean()
    Z = Z / (Z.max() - Z.min())
    Z = Z * amplitude + mean
    return xs, ys, Z


def generate_terrain(world_size, target_resolution, base_period, mean, amplitude, octaves, lacunarity, persistence, seed, record_parameters, file_path):
    """
    Generates terrain using world size in meters and parameters for perlin noise
    and outputs it to a file.

    The files are placed in the model folder. This function uses the
    create_terrain function.

    :param      world_size:         The world size in meters.
    :type       world_size:         float
    :param      target_resolution:  The target raster resolution.
    :type       target_resolution:  float
    :param      base_period:        The period of the base frequency perlin
                                    noise function in meters.
    :type       base_period:        float
    :param      mean:               The mean of the generated bathymetry.
    :type       mean:               float
    :param      amplitude:          The amplitude of the generated bathymetry.
                                    The amplitude is measured as the total
                                    variation of the perlin noise function.
    :type       amplitude:          float
    :param      octaves:            The number of octaves to use when generating
                                    the perlin noise terrain.
    :type       octaves:            int
    :param      lacunarity:         The ratio of successive Perlin noise
                                    function's frequency. Usually set to 2.
    :type       lacunarity:         float
    :param      persistence:        The ratio of successive Perlin noise
                                    function's amplitude. Usually 0.5. Controls
                                    the roughness of the terrain.
    :type       persistence:        float
    :param      seed:               The pseudo-random seed for the perlin noise.
    :type       seed:               int
    :param      record_parameters:  If the parameters for this function should
                                    be recorded in the same folder as the output
                                    file_path.
    :type       record_parameters:  bool
    :param      file_path:          The file path that the generated terrain
                                    should be saved at.
    :type       file_path:          Pathlike
    """
    file_path = Path(file_path).resolve()
    directory = file_path.parent

    parameters = {
        'size': world_size,
        'resolution': target_resolution,
        'base_period' : base_period,
        'mean': mean,
        'amplitude': amplitude,
        'octaves': octaves,
        'lacunarity': lacunarity,
        'persistence': persistence,
        'seed': seed,
    }
    if record_parameters:
        parameters_path = directory / (file_path.stem + '_parameters.json')
        with open(parameters_path, 'w') as outfile:
            json.dump(parameters, outfile)

    x, y, Z = create_terrain(**parameters)
    dst = create_raster(x, y, Z)
    save_raster(dst, file_path)


MeshInfo = namedtuple('MeshInfo', ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'])
RasterInfo = namedtuple('RasterInfo', ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'])


def convert_raster_to_stl(path_tiff, path_stl):
        if path_stl.exists():
            os.remove(path_stl)
        proc = subprocess.run(shlex.split(f'phstl "{path_tiff}" "{path_stl}"'))
        if proc.stderr:
            print(proc.stderr)
            sys.exit(1)


def raster_info(src):
    """
    Get the bounds of the raster.

    :param      src:  The source rasterio raster.
    :type       src:  rasterio dataset
    """
    xmin = min(src.bounds.left, src.bounds.right)
    ymin = min(src.bounds.top, src.bounds.bottom)
    zmin = src.read(1).min()
    xmax = max(src.bounds.left, src.bounds.right)
    ymax = max(src.bounds.top, src.bounds.bottom)
    zmax = src.read(1).max()

    return RasterInfo(
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax
    )


def create_colormap_texture(path_tiff, path_texture, colormap):
    """
    Creates a colormap texture using matplotlib.

    This colormap can be applied as the material used for rendering the final
    terrain.

    :param      path_tiff:     The path tiff
    :type       path_tiff:     { type_description }
    :param      path_texture:  The path texture
    :type       path_texture:  { type_description }
    :param      colormap:      The colormap
    :type       colormap:      { type_description }
    """
    with rio.open(path_tiff, 'r') as src:
        tiff_data = src.read(1)
        tiff_data_extent = rio.plot.plotting_extent(src)

    if colormap is None:
        colormap = 'viridis'

    cmap = matplotlib.cm.get_cmap(colormap)
    norm = plt.Normalize(vmin=tiff_data.min(), vmax=tiff_data.max())
    image = cmap(norm(tiff_data))
    plt.imsave(path_texture, image)


def create_file(path, contents, overwrite=False):
    if overwrite or (not overwrite and not path.exists()):
        with open(path, 'w') as outfile:
            outfile.write(contents)


def magnitude(x):
    return int(math.log10(x))


def create_image_texture_for_blender(raster_path, image_path):
    with rio.open(raster_path) as src:
        arr = src.read(1)
        valid_data = src.read_masks(1) == 255

    arr = rio.fill.fillnodata(arr, valid_data)
    plt.imsave(image_path, arr, cmap='binary_r')

    # size = arr.shape[0]
    # mag = magnitude(size)
    height_extent = arr.max() - arr.min()
    min_height = arr.min()
    print(f'Height extent: {height_extent}')
    print(f'Min height: {min_height}')

    # blender_size = (10**mag*10)
    # print(blender_size)
    # crop_value = (blender_size - size) / blender_size
    crop_value = 0.00
    print(f'crop value {crop_value}')
    strength = 1
    mid_level = -(min_height / abs(height_extent))
    print(f'Mid Level: {mid_level}')
    return mid_level, strength, crop_value


def create_textured_collada_file(image_path, path_collada, path_texture, scale, target_size, target_mesh_resolution, height_exaggeration, height_extent, blender_parameters):
        subdivisions = math.ceil(math.log(target_size/target_mesh_resolution) / math.log(2))
        mid_level, strength, crop_value = blender_parameters
        z_scale = scale * height_extent * height_exaggeration
        print(target_size, target_mesh_resolution, height_exaggeration, height_extent, scale)

        blender_python_script = textwrap.dedent(f"""
        import bpy
        while bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, location=(0, 0, 0))
        mesh = bpy.data.objects[0]

        bpy.ops.object.modifier_add(type='SUBSURF')
        subdivision_modifier = bpy.context.object.modifiers[0]
        subdivision_modifier.subdivision_type = 'SIMPLE'
        subdivision_modifier.render_levels = {subdivisions}
        subdivision_modifier.levels = {subdivisions}
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier=subdivision_modifier.name)

        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.editmode_toggle()

        bpy.ops.object.shade_smooth()

        # The heightmap image
        bpy.ops.image.open(filepath='{image_path}', directory='{image_path.parent}', files=[{{'name':'{image_path.name}', 'name':'{image_path.name}'}}], relative_path=True, show_multiview=False)
        heightmap_image = bpy.data.images['{image_path.name}']

        # The colormap image
        bpy.ops.image.open(filepath='{path_texture}', directory='{path_texture.parent}', files=[{{'name':'{path_texture.name}', 'name':'{path_texture.name}'}}], relative_path=True, show_multiview=False)
        colormap_image = bpy.data.images['{path_texture.name}']

        # Create the material shader node
        bpy.ops.material.new()
        material = bpy.data.materials[0]
        material.name = 'Colormap'
        material_output_node = material.node_tree.nodes['Material Output']
        shader_node = material.node_tree.nodes['Principled BSDF']
        image_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        image_texture_node.image = colormap_image
        image_texture_node.extension = 'EXTEND'
        material.node_tree.links.new(shader_node.inputs[0], image_texture_node.outputs[0])
        mesh.active_material = material

        # Create the displace modifier
        bpy.ops.object.modifier_add(type='DISPLACE')
        displace_mod = bpy.context.object.modifiers['Displace']
        displace_mod.strength = {strength}
        displace_mod.mid_level = {mid_level}
        displace_mod.name = 'Displace'
        displace_mod.direction = 'Z'
        displace_mod.space = 'GLOBAL'

        # Displace the mesh based on the heightmap
        bpy.ops.texture.new()
        displace_image_texture = bpy.data.textures[0]
        displace_image_texture.type = 'IMAGE'
        displace_image_texture.image = heightmap_image
        displace_image_texture.extension = 'EXTEND'
        displace_image_texture.crop_min_x = {crop_value}
        displace_image_texture.crop_min_y = {crop_value}
        displace_image_texture.crop_max_x = {1-crop_value}
        displace_image_texture.crop_max_y = {1-crop_value}
        bpy.context.object.modifiers['Displace'].texture = displace_image_texture
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Displace')

        # Extrude to solidify
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={{'use_normal_flip':False, 'mirror':False}}, TRANSFORM_OT_translate={{'value':(0, 0, -{0.2}), 'orient_type':'GLOBAL', 'orient_matrix':((-0.705802, -0.708354, 0.00887378), (0.70841, -0.705747, 0.00879614), (3.18615e-05, 0.0124946, 0.999922)), 'orient_matrix_type':'NORMAL', 'constraint_axis':(False, False, True), 'mirror':False, 'use_proportional_edit':False, 'proportional_edit_falloff':'SMOOTH', 'proportional_size':1, 'use_proportional_connected':False, 'use_proportional_projected':False, 'snap':False, 'snap_target':'CLOSEST', 'snap_point':(0, 0, 0), 'snap_align':False, 'snap_normal':(0, 0, 0), 'gpencil_strokes':False, 'cursor_transform':False, 'texture_space':False, 'remove_on_cancel':False, 'release_confirm':True, 'use_accurate':False}})
        bpy.ops.object.editmode_toggle()

        # Scale the mesh to the correct size
        bpy.ops.transform.resize(value=({target_size / 2}, {target_size / 2}, {z_scale}), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        mesh.location.x = {target_size / 2}
        mesh.location.y = {target_size / 2}
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Recalculate normals
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()

        # Export the collada mesh
        bpy.ops.wm.collada_export(filepath='{path_collada}', apply_modifiers=True)
        # bpy.ops.export_mesh.stl(filepath='{path_collada.stem + '.stl'}')
        """)

        print('Using blenders python api and running:', blender_python_script)
        # Create textured Collada mesh
        if path_collada.exists():
          os.remove(path_collada)
        proc = subprocess.run(f'blender -b --python-expr "{blender_python_script}"', shell=True, capture_output=True)
        if proc.stderr:
            print(proc.stderr)
            sys.exit(1)

def create_textured_collada_file_old(path_stl, path_collada, path_texture, scale, target_size, target_mesh_resolution, height_exaggeration, buffer_fraction):
        subdivisions = math.ceil(math.log(target_size/target_mesh_resolution) / math.log(2))
        mesh_name = 'object01'
        blender_python_script = textwrap.dedent(f"""
        import bpy
        while bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)

        bpy.ops.import_mesh.stl(filepath='{path_stl}')
        bpy.context.active_object.name = '{mesh_name}'
        bpy.data.objects['{mesh_name}'].rotation_euler[0] = 0
        bpy.data.objects['{mesh_name}'].rotation_euler[1] = 0
        bpy.data.objects['{mesh_name}'].rotation_euler[2] = 0
        bpy.data.objects['{mesh_name}'].scale[0] = {scale}
        bpy.data.objects['{mesh_name}'].scale[1] = {scale}
        bpy.data.objects['{mesh_name}'].scale[2] = {scale*height_exaggeration}

        bpy.ops.preferences.addon_enable(module='io_import_images_as_planes')
        bpy.ops.import_image.to_plane(files=[{{'name':'{path_texture.name}'}}], directory='{path_texture.parent}')
        bpy.data.objects['{path_texture.stem}'].rotation_euler[0] = 0
        bpy.data.objects['{path_texture.stem}'].rotation_euler[1] = 0
        bpy.data.objects['{path_texture.stem}'].rotation_euler[2] = 0
        bpy.data.objects['{path_texture.stem}'].scale[0] = {target_size*(1-2*buffer_fraction)}
        bpy.data.objects['{path_texture.stem}'].scale[1] = {target_size*(1-2*buffer_fraction)}
        bpy.data.objects['{path_texture.stem}'].location[0] = {target_size / 2}
        bpy.data.objects['{path_texture.stem}'].location[1] = {target_size / 2}

        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.context.object.modifiers['Subdivision'].subdivision_type = 'SIMPLE'
        bpy.context.object.modifiers['Subdivision'].render_levels = {subdivisions}
        bpy.context.object.modifiers['Subdivision'].levels = {subdivisions}
        bpy.context.object.modifiers['Subdivision'].quality = 5
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subdivision')

        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.editmode_toggle()

        bpy.ops.object.modifier_add(type='SHRINKWRAP')
        bpy.context.object.modifiers['Shrinkwrap'].target = bpy.data.objects['{mesh_name}']
        bpy.context.object.modifiers['Shrinkwrap'].wrap_method = 'PROJECT'
        bpy.context.object.modifiers['Shrinkwrap'].use_negative_direction = True
        bpy.context.object.modifiers['Shrinkwrap'].use_project_z = True
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Shrinkwrap')

        bpy.ops.object.modifier_add(type='SOLIDIFY')
        bpy.context.object.modifiers['Solidify'].thickness = 0.0001
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Solidify')

        # # TODO: Get rid of this temporary fix.
        # bpy.ops.transform.mirror(orient_type='GLOBAL', constraint_axis=(False, True, False), use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

        bpy.data.objects.remove(bpy.data.objects['{mesh_name}'], do_unlink=True)
        bpy.ops.wm.collada_export(filepath='{path_collada}', apply_modifiers=True)
        """)

        print('Using blenders python api and running:', blender_python_script)
        # Create textured Collada mesh
        if path_collada.exists():
          os.remove(path_collada)
        proc = subprocess.run(f'blender -b --python-expr "{blender_python_script}"', shell=True, capture_output=True)
        if proc.stderr:
            print(proc.stderr)
            sys.exit(1)


def create_model_files(name, path_config, path_sdf, scale, height_exaggeration, mesh_info, path_collada, models_folder, overwrite=False):
    x_extent = mesh_info.xmax - mesh_info.xmin
    y_extent = mesh_info.ymax - mesh_info.ymin
    z_extent = mesh_info.zmax - mesh_info.zmin
    model_config = f"""
    <?xml version="1.0"?>
    <model>
      <name>{name}</name>
      <version>1.0</version>
      <sdf version="1.5">model.sdf</sdf>

      <author>
        <name>{'Franz Heubach'}</name>
        <email>{'franz@heubach.me'}</email>
      </author>

      <description>
        Original mesh scaled by {scale}, with {mesh_info.xmin}, {mesh_info.ymin} at the origin. The height is exaggerated by {height_exaggeration}.
      </description>
    </model>
    """

    model_sdf = f"""
    <?xml version="1.0" ?>
    <sdf version="1.5">
      <model name="{name}">
        <static>true</static>
        <link name="{name}_link">

          <visual name="surface">
            <cast_shadows>false</cast_shadows>
            <transparency>0.4</transparency>
            <pose>{(x_extent * scale) / 2.} {(y_extent * scale) / 2.} 0 0 0 0</pose>
            <geometry>
              <box>
                <size>{x_extent * scale} {y_extent * scale} .1</size>
              </box>
            </geometry>
              <material>
                <script>
                  <uri>file://media/materials/scripts/water.material</uri>
                  <name>UUVSimulator/StaticTurquoiseWater</name>
                </script>
              </material>
          </visual>

          <collision name="{name}_collision">
            <geometry>
              <mesh>
                <uri>model://{path_collada.relative_to(models_folder)}</uri>
              </mesh>
            </geometry>
          </collision>

          <visual name="{name}_visual">
            <geometry>
              <mesh>
                <uri>model://{path_collada.relative_to(models_folder)}</uri>
              </mesh>
            </geometry>
            <material>
            </material>
          </visual>

        </link>
      </model>
    </sdf>
    """
    create_file(path_config, model_config, overwrite=overwrite)
    create_file(path_sdf, model_sdf, overwrite=overwrite)


def create_world_file(name, path_world, overwrite=False):
    world_file = f"""
    <?xml version="1.0" ?>
    <sdf version="1.5">
      <world name="{name}">
        <physics name="default_physics" default="true" type="ode">
          <max_step_size>0.002</max_step_size>
          <real_time_factor>1</real_time_factor>
          <real_time_update_rate>500</real_time_update_rate>
          <ode>
            <solver>
              <type>quick</type>
              <iters>50</iters>
              <sor>0.5</sor>
            </solver>
          </ode>
        </physics>

        <scene>
          <ambient>0.4 0.4 0.4 1</ambient>
          <background>0.7 0.7 0.7 1</background>
          <shadows>false</shadows>
        </scene>

        <light type="directional" name="some_light">
          <diffuse>0.9 0.9 0.9 0</diffuse>
          <specular>1 1 1 0</specular>
          <direction>-1 -1 -1</direction>
        </light>

        <gui>
          <camera name="user_camera">
            <view_controller>orbit</view_controller>
            <pose>137 -1300 444 0 0.3888 1.66</pose>
          </camera>
        </gui>

        <include>
          <uri>model://sun</uri>
        </include>

        <include>
          <uri>model://{name}</uri>
          <pose>0 0 0 0 0 0</pose>
        </include>
      </world>
    </sdf>
    """
    create_file(path_world, world_file, overwrite=overwrite)


def create_open_gazebo_world_script(name, path_open_world_script, ros_gazebo_package_folder, worlds_folder, models_folder, overwrite=False):
    open_world_script = f"""
    source /usr/share/gazebo/setup.sh \
      && GAZEBO_MODEL_PATH="{models_folder}:${{GAZEBO_MODEL_PATH}}" \
      && GAZEBO_RESOURCE_PATH="{worlds_folder}:{ros_gazebo_package_folder}:${{GAZEBO_RESOURCE_PATH}}" \
      && gazebo --verbose {name}.world
    """
    create_file(path_open_world_script, open_world_script, overwrite=overwrite)
    path_open_world_script.chmod(path_open_world_script.stat().st_mode | stat.S_IEXEC)


def terrain_plot(
    x,
    y,
    Z,
    save_figure,
    base_period,
    figure_width_pts=252,
    xlabel='X (m)',
    ylabel='Y (m)',
    zlabel='Depth (m)',
    grid=True,
    xtick_spacing=None,
    ytick_spacing=None):

    plt.rc('font', size=8, family='serif', serif='STIXGeneral')
    pt = 1/72 # in / pt
    width = figure_width_pts*pt
    height = width / 1.3333
    print(f'Figure size (width x height): ({width} in x {height} in), ({width / pt} x {height / pt})')
    plt.figure(figsize=(width, height))

    # xres = x[1] - x[0]
    # yres = x[]

    color = plt.imshow(Z, extent=(x.min()), cmap='binary_r')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(zlabel)
    ax = plt.gca()
    if grid:
        grid = plt.grid(True)
        ax.set_xticks(np.arange(x.min(), x.max()+2, base_period), minor=False)
        ax.set_yticks(np.arange(y.min(), y.max()+2, base_period), minor=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_figure:
        path = Path(save_figure).resolve()
        plt.savefig(path, dpi=300)
        print(f'Figure saved to: {path}')

    plt.show()


def plot_raster(
    src,
    save_figure,
    show_figure,
    base_period,
    figure_width_pts=252,
    xlabel='X (m)',
    ylabel='Y (m)',
    zlabel='Depth (m)',
    grid=True,
    xtick_spacing=None,
    ytick_spacing=None):

    plt.rc('font', size=8, family='serif', serif='STIXGeneral')
    pt = 1/72 # in / pt
    width = figure_width_pts*pt
    height = width / 1.3333
    print(f'Figure size (width x height): ({width} in x {height} in), ({width / pt} x {height / pt})')
    plt.figure(figsize=(width, height))

    xres, yres = src.res

    color = plt.imshow(src.read(1), extent=rio.plot.plotting_extent(src), cmap='binary_r')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(zlabel)
    ax = plt.gca()
    if grid:
        grid = plt.grid(True)
        distortion_ratio = src.res[1] / src.res[0]
        base_period_x = base_period
        base_period_y = distortion_ratio * base_period_x
        xticks = np.arange(src.bounds.left+base_period_x/2, src.bounds.right, base_period_x)
        yticks = np.arange(src.bounds.bottom+base_period_y/2, src.bounds.top, base_period_y)
        ax.set_xticks(xticks, minor=False)
        ax.set_yticks(yticks, minor=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_figure:
        path = Path(save_figure).resolve()
        plt.savefig(path, dpi=300)
        print(f'Figure saved to: {path}')

    if show_figure:
        plt.show()


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def slice_raster(src, n=50, origin=None):
    # Since rasters can have negative resolution we need to deal with that
    # case.
    x_res_negative = src.res[0] < 0
    y_res_negative = src.res[1] < 0

    if origin is None:
        if x_res_negative:
            x_origin = src.bounds.right-src.res[0]/2
        else:
            x_origin = src.bounds.left+src.res[0]/2
        if y_res_negative:
            y_origin = src.bounds.top-src.res[1]/2
        else:
            y_origin = src.bounds.bottom+src.res[1]/2
        origin = (x_origin, y_origin)

    if x_res_negative:
        x_end = src.bounds.left+src.res[0]/2
    else:
        x_end = src.bounds.right-src.res[0]/2
    if y_res_negative:
        y_end = src.bounds.bottom+src.res[1]/2
    else:
        y_end = src.bounds.top-src.res[1]/2

    x_slice = np.linspace(origin[0], x_end, n)
    y_slice = np.linspace(origin[1], y_end, n)

    print(f'Slice: {n} points from {x_slice[0], y_slice[0]} to {x_slice[-1], y_slice[-1]}')

    Z = src.read(1)
    z = np.zeros(n)
    print(f'Shape of raster: {Z.shape}')
    for k in range(n):
        i, j = rio.transform.rowcol(src.transform, x_slice[k], y_slice[k])
        # i = clamp(i, 0, Z.shape[0]-1)
        # j = clamp(j, 0, Z.shape[1]-1)
        z[k] = Z[i, j]

    print(f'Using origin: {origin}')
    distance = [sqrt((x-origin[0])**2 + (y-origin[1])**2) for x, y in zip(np.nditer(x_slice), np.nditer(y_slice))]
    return distance, z


def get_closest_point_index(p, xy):
    diff = np.abs(xy - p)
    distance = np.sum(diff, axis=1)
    return np.argmin(distance)


def slice_stl(m, n=50, origin=None):
    # The mesh x, y, and z values are in groups of three defining the vertices
    # of a face. So the vertices are not unique. We need to extract all the
    # unique ones.
    x_flat = m.x.flatten()
    y_flat = m.y.flatten()
    z_flat = m.z.flatten()
    xy_flat = np.stack((x_flat, y_flat), axis=-1)
    xy, xy_index = np.unique(xy_flat, return_index=True, axis=0)
    z = z_flat[xy_index]
    x, y = np.split(xy, 2, axis=1)
    x, y = x.reshape(-1), y.reshape(-1)

    # Create slice points along the diagonal
    if origin is None:
        origin = (x[0], y[0])

    print(f'Using origin: {origin}')
    x_slice = np.linspace(origin[0], x.max(), n)
    y_slice = np.linspace(origin[1], y.max(), n)
    xy_slice = np.stack((x_slice, y_slice), axis=-1)

    print(f'Slice: {n} points from ({x_slice[0], y_slice[0]}) to ({x_slice[-1], y_slice[-1]})')

    # Calculate the z values closest to the x,y slice points.
    z_slice = np.zeros((xy_slice.shape[0], 1))
    for n in range(xy_slice.shape[0]):
        p = xy_slice[n, :]
        i = get_closest_point_index(p, xy)
        z_slice[n,0] = z[i]

    # Distances along the slice path from the start
    distances = np.sqrt(x_slice**2 + (y_slice**2))
    return distances, z_slice


def slice_collada(m, n=50, origin=None):
    if not m.geometries:
        print('There are no geometries that are part of this collada mesh.')
        return None
    xyz = m.geometries[0].primitives[0].vertex
    if origin is None:
        origin = (xyz[:,0].min(), xyz[:,1].min())
    print(f'Using origin: {origin}')

    x_slice = np.linspace(origin[0], xyz[:,0].max(), n)
    y_slice = np.linspace(origin[1], xyz[:,1].max(), n)
    xy_slice = np.stack((x_slice, y_slice), axis=-1)

    print(f'Slice: {n} points from ({x_slice[0], y_slice[0]}) to ({x_slice[-1], y_slice[-1]})')

    # Calculate the z values closest to the x,y slice points.
    z_slice = np.zeros((xy_slice.shape[0], 1))
    for n in range(xy_slice.shape[0]):
        p = xy_slice[n, :]
        i = get_closest_point_index(p, xyz[:,0:-1])
        z_slice[n,0] = xyz[i,2]

    # Distances along the slice path from the start
    distances = np.sqrt(x_slice**2 + y_slice**2)
    return distances, z_slice
