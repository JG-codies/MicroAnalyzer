import torch as __torch
import os as __os
import numpy as __np
from tqdm.auto import tqdm

from . import preprocessing
from .torch_model import get_instance_segmentation_model
from .utils import merge_detections


def full_segmentation(images: __np.ndarray, model_weights_path, threshold=None, max_detections=100):
    # prepare images for segmentation
    orig_img_h, orig_img_w = images.shape[-2:]
    prepped_images = preprocessing.img_transform(images, as_tensor=True)

    # get model
    device = __torch.device('cuda') if __torch.cuda.is_available() else __torch.device('cpu')
    model = torch_model.get_instance_segmentation_model(2, max_detections)
    model.load_state_dict(__torch.load(__os.path.join(model_weights_path), map_location=device))
    model.eval()

    # extract full, labeled masks and bboxes from prediction
    masks = []
    boxes = []
    for img in tqdm(prepped_images, desc='performing segmentation'):
        # perform segmentation
        with __torch.no_grad():
            pred = model([img])[0]

        img_boxes, img_mask = merge_detections(pred, threshold)

        # crop to original image size
        img_mask = img_mask[:orig_img_h, :orig_img_w]

        # save output data
        masks.append(img_mask)
        boxes.append(img_boxes)

    # stack image lists
    masks = __np.stack(masks)

    return masks, boxes
