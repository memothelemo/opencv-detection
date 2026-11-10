from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
IMAGES_DIR = ROOT_DIR / "images"

# Target image to use to detect road lanes
TARGET_IMAGE = IMAGES_DIR / "straight-lane-2.jpg"

CANNY_LOW = 50
CANNY_HIGH = 150

# Horizon level relative to the image's height (starting from top/0 to bottom/1)
ROI_TOP_RATIO = 0.60

# Fallback for fewer than three markings: center lies within 10% of image width
# on either side of the image midpoint, measured at the lowest detected point.
CENTER_POSITION_TOLERANCE_RATIO = 0.10

# Marking colours in OpenCV HSV (hue 0-179): white paint is unsaturated, yellow
# paint sits below the hue of grass. Tuned on the sample images.
MAX_WHITE_SATURATION = 60
YELLOW_HUE_MIN = 10
YELLOW_HUE_MAX = 28

CONTOUR_EPSILON = 0.75
DASH_ALIGNMENT_TOLERANCE = 8
MERGE_CENTER_DISTANCE = 12
SMOOTHING_RADIUS = 24

MIN_CENTER_HEIGHT = 60
MIN_CONTOUR_LENGTH = 12
MIN_MARKING_CONTRAST = 8
MAX_CENTER_GAP = 20
MAX_DASH_EXTENSION = 40
MAX_DASH_FIT_ERROR = 4
MAX_DASH_GAP = 90
MAX_MARKING_WIDTH = 40
