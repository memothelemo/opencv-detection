import cv2
import numpy as np

from cv2.typing import MatLike
from pathlib import Path

WINDOW_TITLE = "Road Lane Detection Program"
ROI_Y_RATIO = 0.5

# Path to the root repository directory
ROOT_DIR = Path(__file__).parent.parent
IMAGES_DIR = ROOT_DIR / "images"

# Target image to use to detect road lanes
TARGET_IMAGE = IMAGES_DIR / "straight-lane.jpg"

def adjust_gamma(image: MatLike, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Gets the two end points of an average line
def get_average_line_points(lines, y_top, y_bottom):
    if lines is None: return None

    models = []
    weights = []

    for x1, y1, x2, y2 in np.asarray(lines, dtype=float).reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1

        if dy == 0: continue

        a = dx / dy
        b = x1 - a * y1

        models.append((a, b))
        weights.append(np.hypot(dx, dy))

    if not models: return None

    a, b = np.average(models, axis=0, weights=weights)
    top = (round(a * y_top + b), int(y_top))
    bottom = (round(a * y_bottom + b), int(y_bottom))

    return top, bottom

def isolate_left_and_right_lines(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line

        # there's no reason to have the same equal lines
        if x1 == x2: continue

        slope = (y2 - y1) / (x2 - x1)
        if slope < 0: left_lines.append(line)
        else: right_lines.append(line)

    return left_lines, right_lines

# The reason why we need to have a region of interest, while I was reading
# a guide online, because it is expected that an image has clouds and other
# environmental objects that may affect the road detection.
#
# There are many ways to detect but we're focus on this now.
def make_region_of_interest_mask(image: cv2.typing.MatLike):
    height, width = image.shape
    polygon = np.array([[
        (0, height - 1),
        (int(0.20 * width), int(ROI_Y_RATIO * height)),
        (int(0.80 * width), int(ROI_Y_RATIO * height)),
        (width - 1, height - 1),
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)

def detect_road_markings_by_color(image: MatLike) -> MatLike:
    # Convert the image's original color space into HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Look for the white mask that may look like a road because roads
    # may look less saturated among the rest of the picture.
    white_mask = cv2.inRange(
        hsv_image,
        np.array([0, 0, 170], dtype=np.uint8),
        np.array([179, 70, 255], dtype=np.uint8),
    )

    # Expand the white mask edges with image dilation so we
    # can see it clearly.
    near_white = cv2.dilate(
        white_mask,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1
    )

    return near_white

def detect_with_canny(grayscale_image: MatLike) -> MatLike:
    return cv2.Canny(image=grayscale_image, threshold1=80, threshold2=150)

# This is the main entrypoint of the application
def main() -> int:
    image = cv2.imread(TARGET_IMAGE)
    if image is None:
        print(f"[x] Could not find image: {TARGET_IMAGE}!")
        return 1

    # Convert the image into grayscale then blur so we can detect
    # edges much easier. However we need to differentiate colors with
    # the help of image dilation which we will discuss below this group.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(src=grayscale_image, ksize=(5, 5))

    # Use Canny edge detection (reliable way to edge possible road markings)
    roi_mask = make_region_of_interest_mask(grayscale_image)
    canny_edges = detect_with_canny(blurred_image)
    possible_road_edges_from_color = detect_road_markings_by_color(image)

    maybe_road_edges = cv2.bitwise_and(
        cv2.bitwise_and(canny_edges, roi_mask),
        possible_road_edges_from_color
    )

    # Detecting straight lines using the Hough transform
    maybe_road_edges = cv2.HoughLinesP(
        image=maybe_road_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=150,
        maxLineGap=80,
    )

    # HoughLinesP return None if we cannot find one
    if maybe_road_edges is None: maybe_road_edges = []

    # Isolate slanting to the left or to the right lines
    to_left_lines, to_right_lines = isolate_left_and_right_lines(maybe_road_edges)

    image_height = grayscale_image.shape[0]
    left_line_points = get_average_line_points(
        to_left_lines,
        int(image_height * ROI_Y_RATIO),
        image_height
    )
    right_line_points = get_average_line_points(
        to_right_lines,
        int(image_height * ROI_Y_RATIO),
        image_height
    )

    # Rendering the road boundary lines
    for points in (left_line_points, right_line_points):
        if points is None: continue
        start, end = points
        cv2.line(image, start, end, (0, 255, 0), 3)

    # Once we're done, we need to get its middle lane line if we have one.
    # if left_line_points is not None and right_line_points is not None:
    #     # Make a mask that we can only focus on the road based on the boundary only.
    #     left_start, left_end = left_line_points
    #     right_start, right_end = right_line_points

    #     mask_polygon = np.array([[
    #         (left_start[0], left_start[1]),
    #         (left_end[0], left_end[1]),
    #         (right_end[0], right_end[1]),
    #         (right_start[0], right_start[1]),
    #     ]], dtype=np.int32)

    #     mask = np.zeros_like(image)
    #     cv2.fillPoly(mask, mask_polygon, (255, 255, 255))

    #     within_road = cv2.bitwise_and(image, mask)

    #     # Do the thing all over again
    #     grayscale_image = cv2.cvtColor(within_road, cv2.COLOR_BGR2GRAY)
    #     blurred_image = cv2.blur(src=grayscale_image, ksize=(5, 5))
    #     high_contrast_image = adjust_gamma(blurred_image, 0.2)

    #     canny_edges = cv2.Canny(image=high_contrast_image, threshold1=100, threshold2=150)
    #     maybe_lane_edges = cv2.HoughLinesP(
    #         image=canny_edges,
    #         rho=1,
    #         theta=np.pi / 180,
    #         threshold=30,
    #         minLineLength=150,
    #         maxLineGap=80,
    #     )

    #     # HoughLinesP return None if we cannot find one
    #     if maybe_lane_edges is None: maybe_lane_edges = []

    #     # Isolate slanting to the left or to the right lines
    #     to_left_lines, to_right_lines = isolate_left_and_right_lines(maybe_lane_edges)

    #     left_line_points = get_average_line_points(
    #         to_left_lines,
    #         int(image_height * ROI_Y_RATIO),
    #         image_height
    #     )
    #     right_line_points = get_average_line_points(
    #         to_right_lines,
    #         int(image_height * ROI_Y_RATIO),
    #         image_height
    #     )
    #     print(left_line_points)

    #     # Rendering the road lane lines
    #     for points in (left_line_points, right_line_points):
    #         if points is None: continue
    #         start, end = points
    #         cv2.line(image, start, end, (255, 0, 0), 3)

    cv2.imshow(WINDOW_TITLE, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0
