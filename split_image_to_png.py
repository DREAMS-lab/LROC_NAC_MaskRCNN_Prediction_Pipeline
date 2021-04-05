import os
import gdal
from pathlib import Path


def crop_image(input_filename, out_path, count_limit=-1, img_format="PNG", extension=".png"):
    """
        Create a 350x350 cropped tile of the LROC NAC image.

    :param input_filename: NAC image tif file
    :param out_path: folder to store the images
    :param count_limit: limit the number of output images to count_limit, -1 creates all output images
    :param img_format: PNG
    :param extension: .png
    :return:
    """
    print(input_filename, out_path, count_limit)

    output_filename = 'tile_'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    tile_size_x = 350
    tile_size_y = 350

    ds = gdal.Open(input_filename)
    band = ds.GetRasterBand(1)

    xsize = band.XSize
    ysize = band.YSize

    merge_x = 0
    merge_y = 0

    x_inc = tile_size_x
    y_inc = tile_size_y

    count = 0

    for i in range(0, xsize, tile_size_x):

        x_inc = tile_size_x
        # handle corner case
        if i + tile_size_x > xsize:
            i = xsize - tile_size_x

        merge_y = 0
        for j in range(0, ysize, tile_size_y):

            y_inc = tile_size_y

            # handle corner case
            if j + tile_size_y > ysize:
                j = ysize - tile_size_y

            com_string = "gdal_translate -scale -ot Byte -of {} -srcwin ".format(img_format) + str(i) + ", " + str(j) + ", " + str(x_inc) + ", " + str(
                y_inc) + " " + str(input_filename) + " " + str(out_path) + str(output_filename) + str(
                i) + "_" + str(j) + extension

            com_string = f"gdal_translate -scale -ot Byte -of {img_format} " \
                         f"-srcwin {str(i)}, {str(j)}, {str(x_inc)}, {str(y_inc)} {str(input_filename)} " \
                         f"{os.path.join(out_path, output_filename)}{str(i)}_{str(j) + extension}"

            os.system(com_string)
            count += 1
            print("Tile : ", i, ",", j)
            if count == count_limit:
                break
            merge_y = merge_y + tile_size_y
        merge_x = merge_x + tile_size_x
        if count == count_limit:
            break


# Create 5 cropped LROC NAC 350x350 image from the tif file
crop_image(f'NAC_ROI_ALPHNSUSLOA_E129S3581_cropped.tif',
           'predict_images',
           count_limit=5,
           img_format="PNG",
           extension=".png")
