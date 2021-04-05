from PIL import Image
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import os
# from MaskRCNN.visualize import display_instances


def get_model_instance_segmentation(num_classes, image_mean_, image_std_, stats=False):
    """ This function defines the MaskRCNN model

    :param num_classes: number of classes including background
    :param image_mean_: mean from training data
    :param image_std_: std from training data
    :param stats:
    :return:
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # the size shape and the aspect_ratios shape should be the same as the shape in the loaded model
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                       aspect_ratios=(
                                       (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
                                       (0.5, 1.0, 2.0)))
    model.rpn.anchor_generator = anchor_generator

    if stats:
        model.transform.IMAGE_MEAN = image_mean_
        model.transform.IMAGE_STD = image_std_
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.roi_heads.detections_per_img = 256

    return model


def generate_mask(img_path, img_name, mask_rcnn, device, mask_path, display=False):
    """
        Generate MaskRCNN prediction for a single image.
    :param img_path: folder with lroc nac images
    :param img_name: file name of the lroc nac image
    :param mask_rcnn: pytorch maskrcnn model
    :param device: cpu/gpu device object
    :param mask_path: predictions will be stored in this folder
    :param display: show the predictions
    :return:
    """
    image = Image.open(os.path.join(img_path, img_name)).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image / 255.0).float()
    image = image.permute((2, 0, 1))

    pred = mask_rcnn(image.unsqueeze(0).to(device))[0]
    boxes_ = pred["boxes"].cpu().detach().numpy().astype(int)
    boxes = np.empty_like(boxes_)
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = boxes_[:, 1], boxes_[:, 0], boxes_[:, 3], boxes_[:, 2]
    labels = pred["labels"].cpu().detach().numpy()
    scores = pred["scores"].cpu().detach().numpy()
    masks = pred["masks"]

    indices = scores > 0.3  # Adjust this confidence to higher values to only show craters with high confidence
    masks = masks[indices].squeeze(1)
    masks = (masks.permute((1, 2, 0)).cpu().detach().numpy() > 0.5).astype(np.uint8)

    if display:
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        image = image.permute((1, 2, 0)).cpu().detach().numpy() * 255
        # Uncomment this line and install its dependencies (matplotlib) to view the predictions
        # display_instances(image, boxes, masks, labels, class_names=["background", "crater"], scores=scores)

    squashed = masks.any(axis=2)
    new_mask = np.zeros((350, 350))
    new_mask[squashed] = 255
    new_mask[squashed is False] = 0

    im = Image.fromarray(new_mask).convert("L")
    im.save(os.path.join(mask_path, img_name.split(".")[0] + ".png"))


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Following image mean and std from the training data,
# each input image will be normalized to these values
CATEGORY_NAMES = ['background', 'crater']
num_classes = len(CATEGORY_NAMES)
image_mean = [0.34616187074865956, 0.34616187074865956, 0.34616187074865956]
image_std = [0.10754412766560431, 0.10754412766560431, 0.10754412766560431]

# Create the model object, loaded from the weights file
mask_rcnn = get_model_instance_segmentation(num_classes, image_mean, image_std, stats=True)
mask_rcnn.to(device)
mask_rcnn.eval()
model_path = "models/epoch_0009.param"
mask_rcnn.load_state_dict(torch.load(model_path, map_location=device))

IMAGES_DIR = "predict_images"
PREDICTION_DIR = "predicted_masks"

image_files = os.listdir(IMAGES_DIR)
for image_file in image_files:
    if image_file.endswith(".png"):
        generate_mask(IMAGES_DIR, image_file, mask_rcnn, device, PREDICTION_DIR, display=False)
