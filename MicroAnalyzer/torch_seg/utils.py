import numpy as __np


def merge_detections(pred, threshold=None):
    # get mask list and box list
    pred_masks = pred['masks'].to('cpu').numpy()
    pred_boxes = pred['boxes'].to('cpu').numpy()

    # iterate single masks and boxes in and collect those that beat the threshold in confidence.
    # label each mask and put in a single labled mask.
    labeled_mask = __np.zeros(pred_masks.shape[-2:], dtype=__np.integer)
    keep_boxes = []
    missing_count = 0
    prob_map = __np.zeros(pred_masks.shape[-2:])
    for i, (msk, box) in enumerate(zip(pred_masks, pred_boxes)):
        msk = __np.squeeze(msk)

        if threshold is None:
            # if no threshold, create probability maps
            prob_map[msk > 0] = msk[msk > 0]
        else:
            # if threshold given, create labeled masks

            # keep only above the given threshold
            threshold_msk = msk > threshold

            if __np.all(~threshold_msk):
                missing_count += 1
                continue

            # save in full labeled mask
            labeled_mask[threshold_msk] = i - missing_count

        # save this box
        keep_boxes.append(box)

    if threshold is None:
        return keep_boxes, prob_map
    else:
        return keep_boxes, labeled_mask
