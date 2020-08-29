from . import models, preprocessing, losses, metrics, vis, datasets, lr_schedulers, postprocessing
import segutils as __su

import numpy as __np
import inspect as __insp
from tqdm.auto import tqdm as __tqdm


def full_segmentation(images: __np.ndarray, model_config_path: str, model_weights_path, threshold=0.5,
                      preprocessor=preprocessing.Preprocessor, with_probs=False):

    # get model
    model = models.get_model(model_config_path)
    model.load_weights(model_weights_path)

    # prepare images for prediction
    if __insp.isclass(preprocessor):
        preprocessor = preprocessor(model.backbone)
    else:
        preprocessor = preprocessor

    prepped_images = preprocessor.preprocess_batch(images)

    # perform prediction
    predictions = []
    for img in __tqdm(prepped_images, desc='performing segmentation'):
        predictions.append(model.predict(img[None])[0])
    predictions = __np.stack(predictions)

    # reshape output to original images shape
    orig_img_h, orig_img_w = images.shape[1:3]
    predictions = predictions[:, :orig_img_h, :orig_img_w, 0]

    # threshold for binary output
    masks = predictions > threshold

    # label cells and get bboxes
    labeled_masks = __su.imutils.label_mask_stack(masks)
    bboxes = __su.imutils.get_bbox_for_all_objects_in_stack(labeled_masks)

    if with_probs:
        return labeled_masks, bboxes, predictions

    return labeled_masks, bboxes
