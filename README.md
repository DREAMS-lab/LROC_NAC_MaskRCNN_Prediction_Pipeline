# LROC_NAC_MaskRCNN_Prediction_Pipeline
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

