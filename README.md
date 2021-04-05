# LROC NAC MaskRCNN Prediction Pipeline
MaskRCNN prediction for LROC NAC images

**Dependencies**:

1. For running `split_image_to_png.py`, please install GDAL. (The code was run in GDAL version is 3.0.4)

     [GDAL Python Installation Instructions](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html/)


2. For running `predict_lroc.py`, please install NumPy, PyTorch, TorchVision and Pillow.

    [PyTorch Installation Instructions](https://pytorch.org/get-started/locally/)

**Models**:

Download the model from [Google Drive](https://drive.google.com/file/d/1dxYDWb1zZ3vvlUJA11mHxskTfD4hgBJT/view?usp=sharing) and save it in the models folder. 

**How to Run**:

1. `python3 split_image_to_png.py` will split NAC_ROI_ALPHNSUSLOA_E129S3581_cropped.tif into 5 350x350 images and store them in predict_images directory.

2. `python3 predict_lroc.py` will predict on the images inside predict_images directory and output the corresponding PNG masks in predicted_masks folder.


**Example output**:

![image](https://github.com/DREAMS-lab/LROC_NAC_MaskRCNN_Prediction_Pipeline/blob/master/predict_images/tile_350_0.png)
![prediction](https://github.com/DREAMS-lab/LROC_NAC_MaskRCNN_Prediction_Pipeline/blob/master/predicted_masks/tile_350_0.png)



**Convert the predictions to polygons**:
   
   ```gdal_polygonize.py predicted_masks/tile_350_0.png predicted_masks/tile_350_0.geojson -b 1 -f "GeoJSON" out DN ```

**To create output prediction for input tif**:
1. Copy all the *.xml files in `predict_images` to `predicted_masks`, and do a gdal_merge.

   ```gdal_merge.py -o NAC_ROI_ALPHNSUSLOA_E129S3581_predictions.tif predicted_masks/*.png```


