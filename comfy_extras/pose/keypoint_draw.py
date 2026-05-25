"""Pose keypoint drawing primitives shared across pose nodes.

`KeypointDraw` exposes a cv2-or-numpy backend so callers can use one drawing
API regardless of whether OpenCV is installed:

  kd = KeypointDraw()
  kd.draw.circle(canvas, (x, y), radius, color, thickness=-1)
  kd.draw.line(canvas, p1, p2, color, thickness=4)
  kd.draw.fillConvexPoly(canvas, polygon, color)
  kd.draw.ellipse2Poly(center, axes, angle, 0, 360, 1)

It also carries DWPose's body/hand topology + color tables, used by:
  - comfy_extras.nodes_sdpose              (SDPose pose drawing)
  - comfy_extras.pose.export.openpose_2d   (SAM 3D Body 2D pose viz)
  - comfy_extras.pose.export.glb_shared    (SAM 3D Body GLB tables)
"""

import colorsys
import math

import numpy as np


class KeypointDraw:
    """
    Pose keypoint drawing class that supports both numpy and cv2 backends.
    """
    def __init__(self):
        try:
            import cv2
            self.draw = cv2
        except ImportError:
            self.draw = self

        # Hand connections (same for both hands)
        self.hand_edges = [
            [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],      # index
            [0, 9], [9, 10], [10, 11], [11, 12], # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20], # pinky
        ]

        # Body connections - matching DWPose limbSeq (1-indexed, converted to 0-indexed)
        self.body_limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18]
        ]

        # Colors matching DWPose
        self.colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
        ]

    @staticmethod
    def circle(canvas_np, center, radius, color, **kwargs):
        """Draw a filled circle using NumPy vectorized operations."""
        cx, cy = center
        h, w = canvas_np.shape[:2]

        radius_int = int(np.ceil(radius))

        y_min, y_max = max(0, cy - radius_int), min(h, cy + radius_int + 1)
        x_min, x_max = max(0, cx - radius_int), min(w, cx + radius_int + 1)

        if y_max <= y_min or x_max <= x_min:
            return

        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def line(canvas_np, pt1, pt2, color, thickness=1, **kwargs):
        """Draw line using Bresenham's algorithm with NumPy operations."""
        x0, y0, x1, y1 = *pt1, *pt2
        h, w = canvas_np.shape[:2]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err, x, y, line_points = dx - dy, x0, y0, []

        while True:
            line_points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err, x = err - dy, x + sx
            if e2 < dx:
                err, y = err + dx, y + sy

        if thickness > 1:
            radius, radius_int = (thickness / 2.0) + 0.5, int(np.ceil((thickness / 2.0) + 0.5))
            for px, py in line_points:
                y_min, y_max, x_min, x_max = max(0, py - radius_int), min(h, py + radius_int + 1), max(0, px - radius_int), min(w, px + radius_int + 1)
                if y_max > y_min and x_max > x_min:
                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                    canvas_np[y_min:y_max, x_min:x_max][(xx - px)**2 + (yy - py)**2 <= radius**2] = color
        else:
            line_points = np.array(line_points)
            valid = (line_points[:, 1] >= 0) & (line_points[:, 1] < h) & (line_points[:, 0] >= 0) & (line_points[:, 0] < w)
            if (valid_points := line_points[valid]).size:
                canvas_np[valid_points[:, 1], valid_points[:, 0]] = color

    @staticmethod
    def fillConvexPoly(canvas_np, pts, color, **kwargs):
        """Fill polygon using vectorized scanline algorithm."""
        if len(pts) < 3:
            return
        pts = np.array(pts, dtype=np.int32)
        h, w = canvas_np.shape[:2]
        y_min, y_max, x_min, x_max = max(0, pts[:, 1].min()), min(h, pts[:, 1].max() + 1), max(0, pts[:, 0].min()), min(w, pts[:, 0].max() + 1)
        if y_max <= y_min or x_max <= x_min:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)

        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            y1, y2 = p1[1], p2[1]
            if y1 == y2:
                continue
            if y1 > y2:
                p1, p2, y1, y2 = p2, p1, p2[1], p1[1]
            if not (edge_mask := (yy >= y1) & (yy < y2)).any():
                continue
            mask ^= edge_mask & (xx >= p1[0] + (yy - y1) * (p2[0] - p1[0]) / (y2 - y1))

        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1, **kwargs):
        """Python implementation of cv2.ellipse2Poly."""
        axes = (axes[0] + 0.5, axes[1] + 0.5) # to better match cv2 output
        angle = angle % 360
        if arc_start > arc_end:
            arc_start, arc_end = arc_end, arc_start
        while arc_start < 0:
            arc_start, arc_end = arc_start + 360, arc_end + 360
        while arc_end > 360:
            arc_end, arc_start = arc_end - 360, arc_start - 360
        if arc_end - arc_start > 360:
            arc_start, arc_end = 0, 360

        angle_rad = math.radians(angle)
        alpha, beta = math.cos(angle_rad), math.sin(angle_rad)
        pts = []
        for i in range(arc_start, arc_end + delta, delta):
            theta_rad = math.radians(min(i, arc_end))
            x, y = axes[0] * math.cos(theta_rad), axes[1] * math.sin(theta_rad)
            pts.append([int(round(center[0] + x * alpha - y * beta)), int(round(center[1] + x * beta + y * alpha))])

        unique_pts, prev_pt = [], (float('inf'), float('inf'))
        for pt in pts:
            if (pt_tuple := tuple(pt)) != prev_pt:
                unique_pts.append(pt)
                prev_pt = pt_tuple

        return unique_pts if len(unique_pts) > 1 else [[center[0], center[1]], [center[0], center[1]]]

    def draw_wholebody_keypoints(self, canvas, keypoints, scores=None, threshold=0.3,
                                 draw_body=True, draw_feet=True, draw_face=True, draw_hands=True,
                                 stick_width=4, face_point_size=3,
                                 marker_radius=4, hand_stick_width=2, hand_marker_radius=4,
                                 limb_alpha=1.0, hand_dot_color=(0, 0, 255)):
        """
        Draw wholebody keypoints (134 keypoints after processing) in DWPose style.

        Expected keypoint format (after neck insertion and remapping):
        - Body: 0-17 (18 keypoints in OpenPose format, neck at index 1)
        - Foot: 18-23 (6 keypoints)
        - Face: 24-91 (68 landmarks)
        - Right hand: 92-112 (21 keypoints)
        - Left hand: 113-133 (21 keypoints)

        Args:
            canvas: The canvas to draw on (numpy array)
            keypoints: Array of keypoint coordinates
            scores: Optional confidence scores for each keypoint
            threshold: Minimum confidence threshold for drawing keypoints
            stick_width: Body limb half-width (passed to ellipse2Poly).
            face_point_size: Radius of the white face dots.
            marker_radius: Radius of body/foot dots. Defaults to 4 (DWPose).
            hand_stick_width: Thickness of hand limb lines. Defaults to 2.
            hand_marker_radius: Radius of hand dots. Defaults to 4.
            limb_alpha: Body-limb alpha blend (0..1). 1.0 = opaque fill (default),
                <1.0 enables per-limb bbox-clipped alpha overlay (DWPose semantics
                where overlapping limbs darken).
            hand_dot_color: Either an (R, G, B) tuple/list of ints for solid-color
                hand dots (default (0, 0, 255), DWPose blue), or a (21, 3) array
                for per-keypoint hand-dot colors (OpenPose-style rainbow palette).

        Returns:
            canvas: The canvas with keypoints drawn
        """
        H, W, C = canvas.shape

        # Normalize hand_dot_color to a (21, 3) int array.
        hdc_arr = np.asarray(hand_dot_color, dtype=int)
        if hdc_arr.ndim == 1:
            hdc_arr = np.tile(hdc_arr.reshape(1, 3), (21, 1))
        hand_dot_tuples = [tuple(int(c) for c in hdc_arr[i]) for i in range(21)]

        do_alpha = float(limb_alpha) < 1.0

        # Draw body limbs
        if draw_body and len(keypoints) >= 18:
            for i, limb in enumerate(self.body_limbSeq):
                # Convert from 1-indexed to 0-indexed
                idx1, idx2 = limb[0] - 1, limb[1] - 1

                if idx1 >= 18 or idx2 >= 18:
                    continue

                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                Y = [keypoints[idx1][0], keypoints[idx2][0]]
                X = [keypoints[idx1][1], keypoints[idx2][1]]
                mX, mY = (X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2
                length = math.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2)

                if length < 1:
                    continue

                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

                polygon = self.draw.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stick_width), int(angle), 0, 360, 1)

                color = self.colors[i % len(self.colors)]
                if do_alpha:
                    _fill_poly_alpha(canvas, polygon, color, limb_alpha, self.draw)
                else:
                    self.draw.fillConvexPoly(canvas, polygon, color)

        # Draw body keypoints
        if draw_body and len(keypoints) >= 18:
            for i in range(18):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), marker_radius, self.colors[i % len(self.colors)], thickness=-1)

        # Draw foot keypoints (18-23, 6 keypoints)
        if draw_feet and len(keypoints) >= 24:
            for i in range(18, 24):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), marker_radius, self.colors[i % len(self.colors)], thickness=-1)

        # Draw right hand (92-112)
        if draw_hands and len(keypoints) >= 113:
            eps = 0.01
            for ie, edge in enumerate(self.hand_edges):
                idx1, idx2 = 92 + edge[0], 92 + edge[1]
                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])

                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        # HSV to RGB conversion for rainbow colors
                        r, g, b = colorsys.hsv_to_rgb(ie / float(len(self.hand_edges)), 1.0, 1.0)
                        color = (int(r * 255), int(g * 255), int(b * 255))
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=hand_stick_width)

            # Draw right hand keypoints
            for i in range(92, 113):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), hand_marker_radius, hand_dot_tuples[i - 92], thickness=-1)

        # Draw left hand (113-133)
        if draw_hands and len(keypoints) >= 134:
            eps = 0.01
            for ie, edge in enumerate(self.hand_edges):
                idx1, idx2 = 113 + edge[0], 113 + edge[1]
                if scores is not None:
                    if scores[idx1] < threshold or scores[idx2] < threshold:
                        continue

                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])

                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        # HSV to RGB conversion for rainbow colors
                        r, g, b = colorsys.hsv_to_rgb(ie / float(len(self.hand_edges)), 1.0, 1.0)
                        color = (int(r * 255), int(g * 255), int(b * 255))
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=hand_stick_width)

            # Draw left hand keypoints
            for i in range(113, 134):
                if scores is not None and i < len(scores) and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), hand_marker_radius, hand_dot_tuples[i - 113], thickness=-1)

        # Draw face keypoints (24-91) - white dots only, no lines
        if draw_face and len(keypoints) >= 92:
            eps = 0.01
            for i in range(24, 92):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), face_point_size, (255, 255, 255), thickness=-1)

        return canvas


def _fill_poly_alpha(canvas, polygon, color, alpha, draw_backend):
    """Bbox-clipped alpha-blended fillConvexPoly. `canvas` is mutated in-place.

    DWPose semantics: each limb blends with `alpha` independently so overlapping
    limbs darken further. Operates on the polygon's bbox to avoid copying the
    whole canvas per limb.
    """
    H, W = canvas.shape[:2]
    poly_arr = np.asarray(polygon, dtype=np.int32)
    x0 = max(0, int(poly_arr[:, 0].min()))
    xN = min(W, int(poly_arr[:, 0].max()) + 1)
    y0 = max(0, int(poly_arr[:, 1].min()))
    yN = min(H, int(poly_arr[:, 1].max()) + 1)
    if xN <= x0 or yN <= y0:
        return
    local_poly = poly_arr - np.array([x0, y0], dtype=poly_arr.dtype)
    roi = canvas[y0:yN, x0:xN].copy()
    draw_backend.fillConvexPoly(roi, local_poly, color)
    a = float(alpha)
    canvas[y0:yN, x0:xN] = np.clip(
        roi.astype(np.float32) * a + canvas[y0:yN, x0:xN].astype(np.float32) * (1.0 - a),
        0, 255,
    ).astype(np.uint8)
