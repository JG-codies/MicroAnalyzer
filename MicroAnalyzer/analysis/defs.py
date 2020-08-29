BINARY_CHANNEL_KEY = 'binary'
BRIGHTFIELD_CHANNEL_KEY = 'brightfield'
FLUORESCENT_CHANNEL_KEY = 'fluorescence'

REQUIRED_KEYS = [BINARY_CHANNEL_KEY, BRIGHTFIELD_CHANNEL_KEY]

POLARITY_FACTOR = 0.25
RADIUS_MEASURE_MODE = 'mid'

HORIZONTAL_AXIS = 'h'
VERTICAL_AXIS = 'v'
ALL_AXES = 'hv'

FLU_MASK_NAME_TEMPLATE = '{}_mask'

# CSV output name
CSV_OUTPUT_NAME = 'database.csv'

# image visual outputs directory template
IMAGE_DIR_TEMPLATE = 'img_{:06d}'
IMG_EXT = 'tif'

CLUSTER_3D_HIST_DIR_NAME = 'clusters 3D histograms'

# vis
CELL_BBOX_COLOR = 'red'
FLU_BBOX_COLOR = 'green'
FLU_BBOX_PADDING = 5


class DBKeys:
    # cell properties
    IMG_ID = 'frame id'
    ID = 'cell id'
    LENGTH = 'length'
    WIDTH = 'width'
    AREA = 'area'
    RADIUS = 'radius'
    CIRCUMFERENCE = 'circumference'
    SURFACE_AREA = 'surface area'
    VOLUME = 'volume'

    # fluorescent properties
    CELL_MEAN_INTENSITY = 'cell mean intensity'
    CELL_STD_INTENSITY = 'cell std intensity'
    CELL_INTENSITY_CVI = 'cell intensity CVI'
    SUM_INTENSITY_PROFILE_H = 'horizontal sum intensity profile'
    SUM_INTENSITY_PROFILE_V = 'vertical sum intensity profile'
    MEAN_INTENSITY_PROFILE_H = 'horizontal mean intensity profile'
    MEAN_INTENSITY_PROFILE_V = 'vertical mean intensity profile'
    MAX_INTENSITY_PROFILE_H = 'horizontal max intensity profile'
    MAX_INTENSITY_PROFILE_V = 'vertical max intensity profile'

    # cluster properties
    NUM_CLUSTERS = 'number of clusters'
    HAS_CLUSTERS = 'has clusters'
    CLUSTER = 'cluster'
    LEADING_CLUSTER_IDX = 'leading cluster index'
    CLUSTER_ID = 'cluster id'
    SIZE = 'size'
    MEAN_INTENSITY = 'mean intensity'
    STD_INTENSITY = 'std intensity'
    MAX_INTENSITY = 'max intensity'
    SUM_INTENSITY = 'sum intensity'
    CLUSTER_CENTER = 'center'
    IS_POLAR = 'is polar'
