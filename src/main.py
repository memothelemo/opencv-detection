import cv2

from pathlib import Path


WINDOW_TITLE = "Road Lane Detection Program"

# Path to the root repository directory
ROOT_DIR = Path(__file__).parent.parent
IMAGES_DIR = ROOT_DIR / "images"

# Target image to use to detect road lanes
TARGET_IMAGE = IMAGES_DIR / "straight-lane.jpg"


# This is the main entrypoint of the application
def main() -> int:
    image = cv2.imread(TARGET_IMAGE)
    if image is None:
        print(f"[x] Could not find image: {TARGET_IMAGE}!")
        return 1

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_TITLE, image)
    cv2.waitKey(0)
    return 0
