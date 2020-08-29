from segmentation_models import get_preprocessing as __get_prep


def get_preprocessing_func_by_backbone(backbone):
    return __get_prep(backbone.value)
