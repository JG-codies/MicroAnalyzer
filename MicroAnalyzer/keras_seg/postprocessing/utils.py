import mahotas as mh
import numpy as np


def label_mask(binary_mask):
    labeled_msk, _ = mh.label(binary_mask)
    return labeled_msk


def label_mask_stack(binary_stack):
    """
    label a binary stack
    :param binary_stack:
    :return:
    """
    labeled_masks = []
    for img in binary_stack:
        labeled_masks.append(label_mask(img))

    return np.stack(labeled_masks)


def get_bboxes(labeled_mask):
    # instances are encoded as different colors
    obj_ids = np.unique(labeled_mask)

    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # separate all masks into individual masks
    masks = labeled_mask == obj_ids[:, None, None]

    boxes = []
    for i in range(len(obj_ids)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        boxes.append([xmin, ymin, xmax, ymax])

    return boxes


def get_stack_bboxes(labeled_mask_stack):
    boxes = []
    for msk in labeled_mask_stack:
        boxes.append(get_bboxes(msk))

    return boxes
