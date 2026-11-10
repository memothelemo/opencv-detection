import cv2
import numpy as np

from cv2.typing import MatLike
from pathlib import Path

from src import config

ROOT_DIR = Path(__file__).parent.parent
IMAGES_DIR = ROOT_DIR / "images"

# Target image to use to detect road lanes
TARGET_IMAGE = IMAGES_DIR / "curves.jpg"

def main() -> int:
    image = cv2.imread(TARGET_IMAGE)
    if image is None:
        print(f"[x] Could not find image: {TARGET_IMAGE}!")
        return 1

    cv2.imshow("Road Detection System", get_road_boundaries(image))
    cv2.waitKey(0)
    return 0

# Determines the boundaries of a road.
def get_road_boundaries(
    image: MatLike,
    grayscale_image: MatLike | None = None
):
    if image is None:
        raise ValueError("An input image is required.")

    # Blur before Canny to smoothen out some noise
    if grayscale_image is None:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif grayscale_image.shape != image.shape[:2]:
        raise ValueError("The grayscale image must match the input height and width.")

    blurred_gray_image = cv2.blur(src=grayscale_image, ksize=(5, 5))

    # Adjust contrast before detecting edges.
    corrected_image = correct_gamma(blurred_gray_image, gamma=1.5)

    # Use Canny edge detection (reliable way to edge possible road markings)
    canny_edges = cv2.Canny(
        image=corrected_image,
        threshold1=config.CANNY_LOW,
        threshold2=config.CANNY_HIGH
    )
    roi_mask = make_contour_roi(canny_edges)
    maybe_road_edges = cv2.bitwise_and(canny_edges, roi_mask)

    # Another way to detect lane markings is to find contours, because road
    # markings have their own distinct colors.
    contours, _ = cv2.findContours(maybe_road_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour_edges = np.zeros_like(maybe_road_edges)
    for contour in contours:
        if cv2.arcLength(contour, closed=True) < config.MIN_CONTOUR_LENGTH:
            continue

        cv2.drawContours(contour_edges, [contour], -1, 255, 1)

    # Average out the contour edges into a single flowing line (possibly)
    # Paint is white (unsaturated) or yellow; this rejects grass and foliage.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    marking_mask = cv2.inRange(hsv, (0, 0, 0), (179, config.MAX_WHITE_SATURATION, 255))
    marking_mask |= cv2.inRange(
        hsv, (config.YELLOW_HUE_MIN, 0, 0), (config.YELLOW_HUE_MAX, 255, 255)
    )

    left, center, right = get_lane_centerlines(contour_edges, blurred_gray_image, marking_mask)
    output = image.copy()

    # Draw every segment of each polyline, not just the first, so curves are followed.
    for line, color in ((left, (0, 255, 0)), (right, (0, 255, 0)), (center, (0, 0, 255))):
        if line is not None:
            cv2.polylines(output, [line], isClosed=False, color=color, thickness=3)

    # It must not draw above the ROI boundary either.
    top = int(image.shape[0] * config.ROI_TOP_RATIO)
    output[:top] = image[:top]

    return output

# Track the midpoint between nearby edges from the bottom of the image upward.
def get_lane_centerlines(
    contour_edges: MatLike, grayscale_image: MatLike, marking_mask: MatLike | None = None
) -> tuple[MatLike | None, MatLike | None, MatLike | None]:
    """Return (left, center, right) polylines, with None for missing markings.

    If given, marking_mask is nonzero where the pixel colour could be road paint.
    """
    height, width = contour_edges.shape
    top = int(height * config.ROI_TOP_RATIO)
    tracks = []

    for y in range(height - 2, top, -2):
        pixels = np.flatnonzero(contour_edges[y])
        runs = np.split(pixels, np.where(np.diff(pixels) > 1)[0] + 1)
        edges = [int(round(np.mean(run))) for run in runs if len(run)]

        centers = []
        for left, right in zip(edges, edges[1:]):
            if not 2 <= right - left <= config.MAX_MARKING_WIDTH:
                continue
            if left < 4 or right + 4 >= width:
                continue

            # Avoid averaging across a dark gap between separate markings.
            middle = (left + right) // 2
            if marking_mask is not None and not marking_mask[y, middle]:
                continue
            inside = float(grayscale_image[y, middle])
            outside = max(float(grayscale_image[y, left - 4]), float(grayscale_image[y, right + 4]))
            contrast = inside - outside

            if contrast >= config.MIN_MARKING_CONTRAST:
                centers.append((left + right) / 2)

        # Collapse close center candidates to one representative position.
        # The full cluster width is bounded so distant lanes cannot chain together.
        clusters = []
        for x in centers:
            if not clusters or x - clusters[-1][0] > config.MERGE_CENTER_DISTANCE:
                clusters.append([x])
            else:
                clusters[-1].append(x)
        centers = [float(np.mean(cluster)) for cluster in clusters]

        matches = []
        for track_index, track in enumerate(tracks):
            gap = track[-1][1] - y
            if gap > config.MAX_CENTER_GAP:
                continue

            # Predict the next position from the recent direction of the track.
            recent = track[-6:]
            slope = 0.0
            if len(recent) > 1:
                slope = (recent[-1][0] - recent[0][0]) / (recent[-1][1] - recent[0][1])
            predicted_x = track[-1][0] - gap * slope

            for center_index, x in enumerate(centers):
                distance = abs(predicted_x - x)
                if distance <= max(6, gap):
                    matches.append((distance, track_index, center_index))

        # Give each center to at most one track, starting with the closest match.
        used_tracks, used_centers = set(), set()
        for _, track_index, center_index in sorted(matches):
            if track_index not in used_tracks and center_index not in used_centers:
                tracks[track_index].append((centers[center_index], y))
                used_tracks.add(track_index)
                used_centers.add(center_index)
        for center_index, x in enumerate(centers):
            if center_index not in used_centers:
                tracks.append([(x, y)])

    centerlines = []
    for track, joined_dashes in join_dashed_tracks(tracks):
        if len(track) < 10 or track[0][1] - track[-1][1] < config.MIN_CENTER_HEIGHT:
            continue

        # Fit locally so distant bends do not pull the bottom endpoint sideways.
        # Use the original samples for every fit, including one-sided endpoint fits.
        points = np.array(track, dtype=np.float32)
        if joined_dashes:
            # A verified straight sequence can bridge the gaps between dashes.
            # Beyond the last observed dash, this is an estimate, not a detection.
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
            end_y = int(points[-1, 1])
            if end_y - top <= config.MAX_DASH_EXTENSION and 0 <= slope * top + intercept < width:
                end_y = top
            ys = np.append(np.arange(points[0, 1], end_y, -2), end_y)
            points = np.column_stack((slope * ys + intercept, ys)).astype(np.float32)

        original_x = points[:, 0].copy()
        for index, (_, y) in enumerate(points):
            offsets = points[:, 1] - y
            nearby = np.abs(offsets) <= config.SMOOTHING_RADIUS
            if np.count_nonzero(nearby) < 3:
                continue
            coefficients = np.polyfit(
                offsets[nearby] / config.SMOOTHING_RADIUS, original_x[nearby], 2
            )
            points[index, 0] = np.clip(coefficients[-1], 0, width - 1)

        # Geometric detection: represent the smooth center as straight segments.
        polygon = cv2.approxPolyDP(
            points.reshape(-1, 1, 2),
            epsilon=config.CONTOUR_EPSILON,
            closed=False
        )
        centerlines.append(np.rint(polygon).astype(np.int32))

    return isolate_lane_markings(centerlines, width)


def isolate_lane_markings(
    centerlines: list[MatLike], image_width: int
) -> tuple[MatLike | None, MatLike | None, MatLike | None]:
    """Order markings by their lowest observed point, preserving curved lines.

    With three or more candidates, the outermost lines define the boundaries;
    the interior line with the greatest vertical span represents the center.
    With fewer candidates, road-relative positions are ambiguous: use the
    image center band as a fallback and leave missing slots as None.
    """
    def bottom_x(line):
        points = line.reshape(-1, 2)
        return int(points[np.argmax(points[:, 1]), 0])

    def vertical_span(line):
        return int(np.ptp(line.reshape(-1, 2)[:, 1]))

    ordered = sorted(centerlines, key=bottom_x)
    if len(ordered) >= 3:
        left, right = ordered[0], ordered[-1]
        road_middle = (bottom_x(left) + bottom_x(right)) / 2
        center = max(
            ordered[1:-1],
            key=lambda line: (vertical_span(line), -abs(bottom_x(line) - road_middle)),
        )
        return left, center, right

    selected = [None, None, None]
    image_middle = (image_width - 1) / 2
    center_tolerance = image_width * config.CENTER_POSITION_TOLERANCE_RATIO
    for line in ordered:
        x = bottom_x(line)
        if abs(x - image_middle) <= center_tolerance:
            slot = 1
        else:
            slot = 0 if x < image_middle else 2
        if selected[slot] is None or vertical_span(line) > vertical_span(selected[slot]):
            selected[slot] = line

    return selected[0], selected[1], selected[2]

# Join separate dashes only when all their centers support the same straight line
def join_dashed_tracks(tracks):
    ordered = sorted(tracks, key=lambda track: track[0][1], reverse=True)
    used = set()
    groups = []

    for index, track in enumerate(ordered):
        if index in used: continue

        used.add(index)
        merged = list(track)
        joined = False

        # A long initial section makes a direction
        if len(merged) >= 10 and merged[0][1] - merged[-1][1] >= config.MIN_CENTER_HEIGHT:
            # Stop extending this lane without stopping the outer track loop.
            while True:
                points = np.asarray(merged, dtype=float)
                model = np.polyfit(points[:, 1], points[:, 0], 1)
                dash_fit_error = np.max(np.abs(np.polyval(model, points[:, 1]) - points[:, 0]))

                if dash_fit_error > config.MAX_DASH_FIT_ERROR:
                    break

                matches = []
                for candidate_index, candidate in enumerate(ordered):
                    if candidate_index in used or len(candidate) < 3:
                        continue

                    center_gap = merged[-1][1] - candidate[0][1]
                    if not config.MAX_CENTER_GAP < center_gap <= config.MAX_DASH_GAP:
                        continue

                    candidate_points = np.asarray(candidate, dtype=float)
                    errors = np.abs(np.polyval(model, candidate_points[:, 1]) - candidate_points[:, 0])

                    if np.max(errors) > config.DASH_ALIGNMENT_TOLERANCE:
                        continue

                    combined = np.vstack((points, candidate_points))
                    fitted = np.polyfit(combined[:, 1], combined[:, 0], 1)
                    residuals = np.abs(np.polyval(fitted, combined[:, 1]) - combined[:, 0])

                    if np.max(residuals) <= config.MAX_DASH_FIT_ERROR:
                        matches.append((float(np.mean(errors)), center_gap, candidate_index))

                if not matches:
                    break

                _, _, best_index = min(matches)
                merged.extend(ordered[best_index])
                used.add(best_index)
                joined = True

        groups.append((merged, joined))

    return groups

def make_contour_roi(canny_edges: MatLike) -> MatLike:
    mask = np.zeros_like(canny_edges)
    top = int(canny_edges.shape[0] * config.ROI_TOP_RATIO)
    mask[top:, :] = 255
    return mask

# Adjusts the image's gamma
def correct_gamma(image: MatLike, gamma = 1.0):
    return np.array(255 * (image / 255) ** gamma, dtype='uint8')
