import os

__script_dir = os.path.dirname(__file__)

CELL_SEGMENTATION_MODEL_WEIGHTS = os.path.join(__script_dir, 'resources', 'models', 'weights',
                                               'cells_mask_rcnn_torch.pkl')

CLUSTER_SEGMENTATION_MODEL_CONFIG = os.path.join(__script_dir, 'resources', 'models', 'config', 'fluo_fpn.yaml')
CLUSTER_SEGMENTATION_MODEL_WEIGHTS = os.path.join(__script_dir, 'resources', 'models', 'weights',
                                                  'fluo_fpn_keras.hd5')

CELL_SEGMENTATION_MODEL_ID = '1wWt-e0RP0uYy8FLYVS-rUeYER9H2V1BH'
CLUSTER_SEGMENTATION_MODEL_ID = '1DH3RoSh_AowcS1Tc5NdOTfkMgiHliDgB'

