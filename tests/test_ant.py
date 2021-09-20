import ant
import random
import rasterio.plot
import rasterio as rio
import matplotlib.pyplot as plt
import math


EPSILON = 1e-5
TEST_RASTER = '/home/franz/Documents/Masters/workspace/ros1_ws/src/ds_uuv_nav/ds_gazebo_worlds/models/gebco_03-generation/gebco_03.tif'


def almost_equal(a, b):
    print(a,b)
    return abs(a - b) < EPSILON


def test_extract_arrays_write_and_load():
    # Reading the raster, then extracting arrays, then saving, then reading
    # should give the same array and transform.
    with rio.open(TEST_RASTER) as src:
        x_jvec, y_ivec, Z = ant.extract_arrays_from_raster(src)
        x, y = 739716.5550640664, 4227201.467485287
        i, j = rio.transform.rowcol(src.transform, x, y)
        print(i, j)
        print(x, y)
        print(x_jvec[j], y_ivec[i])
        assert almost_equal(x_jvec[j], x) or 'X lookup check fails'
        assert almost_equal(y_ivec[i], y) or 'Y lookup check fails'

        raster = ant.create_raster(x_jvec, y_ivec, Z)
        assert src.transform.almost_equals(raster.transform) or not 'Original transform and created transform is not equal'


def test_interpolate_raster_function():
    with rio.open(TEST_RASTER) as src:
        interpolate_raster = ant.prepare_interpolation_function(src)
        x_jvec, y_ivec, Z = ant.extract_arrays_from_raster(src)
        print('Compare raster value with interpolated value.')
        for i in range(10):
            i, j = random.choice(range(src.height)), random.choice(range(src.width))
            x, y = x_jvec[j], y_ivec[i]
            assert almost_equal(Z[i,j], interpolate_raster((x,y))[0]) or not 'Interpolated value not equal to original raster value'
            print(f'Z at (x={x:.2f}, y={y:.2f}): {Z[i, j]:.2f} == {interpolate_raster((x,y))[0]:.2f} ... Pass')


def test_resample_raster():
    x_resolution_factor = 1/10
    y_resolution_factor = 1/5
    with rio.open(TEST_RASTER) as src:
        dataset = ant.resample_raster(src, x_resolution_factor, y_resolution_factor)
        print(f'Source resolution: {src.res}')
        print(f'Resampled resolution: {dataset.res}')
        actual_x_resolution_factor = dataset.res[0] / src.res[0]
        actual_y_resolution_factor = dataset.res[1] / src.res[1]

        assert almost_equal(round(actual_x_resolution_factor, 2), x_resolution_factor) or not 'Resampling failed to generate the correct resolution scaling'
        assert almost_equal(round(actual_y_resolution_factor, 2), y_resolution_factor) or not 'Resampling failed to generate the correct resolution scaling'


def test_crop_raster_to_square():
    with rio.open(TEST_RASTER) as src:
        rio.plot.show(src)
        dataset = ant.crop_raster_to_square(src)
        plt.imshow(dataset.read(1), extent=rio.plot.plotting_extent(src))
        plt.show()


def test_affine_transformation():
    with rio.open(TEST_RASTER) as src:
        x, y = src.transform * (0,src.height)
        print(x,y)

def test_octaves_from_smallest_feature():
    octaves = ant.octaves_from_smallest_feature(size=5, base_period=100, lacunarity=2)
    assert octaves == 6

def main():
    test_extract_arrays_write_and_load()
    test_interpolate_raster_function()
    test_resample_raster()
    test_crop_raster_to_square()
    test_affine_transformation()
    test_octaves_from_smallest_feature()


if __name__ == '__main__':
    main()



