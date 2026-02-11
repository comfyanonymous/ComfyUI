import torch
import comfy.utils
import numpy as np
import math
import colorsys
from tqdm import tqdm
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_lotus import LotusConditioning


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
                                 draw_body=True, draw_feet=True, draw_face=True, draw_hands=True):
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

        Returns:
            canvas: The canvas with keypoints drawn
        """
        H, W, C = canvas.shape
        stickwidth = 4

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

                polygon = self.draw.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

                self.draw.fillConvexPoly(canvas, polygon, self.colors[i % len(self.colors)])

        # Draw body keypoints
        if draw_body and len(keypoints) >= 18:
            for i in range(18):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, self.colors[i % len(self.colors)], thickness=-1)

        # Draw foot keypoints (18-23, 6 keypoints)
        if draw_feet and len(keypoints) >= 24:
            for i in range(18, 24):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, self.colors[i % len(self.colors)], thickness=-1)

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
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

            # Draw right hand keypoints
            for i in range(92, 113):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

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
                        self.draw.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

            # Draw left hand keypoints
            for i in range(113, 134):
                if scores is not None and i < len(scores) and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

        # Draw face keypoints (24-91) - white dots only, no lines
        if draw_face and len(keypoints) >= 92:
            eps = 0.01
            for i in range(24, 92):
                if scores is not None and scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                    self.draw.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

        return canvas

class SDPoseDrawKeypoints(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDPoseDrawKeypoints",
            category="image/preprocessors",
            search_aliases=["openpose", "pose detection", "preprocessor", "keypoints", "sdpose"],
            inputs=[
                io.Custom("POSEKEYPOINTS").Input("keypoints"),
                io.Boolean.Input("draw_body", default=True),
                io.Boolean.Input("draw_feet", default=True),
                io.Boolean.Input("draw_face", default=True),
                io.Boolean.Input("draw_hands", default=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, keypoints, draw_body, draw_feet, draw_face, draw_hands) -> io.NodeOutput:

        all_keypoints = keypoints["keypoints"]
        all_scores = keypoints["scores"]
        height, width = keypoints["image_size"]

        # Convert all keypoints and scores to numpy upfront
        keypoints_np_batch = []
        scores_np_batch = []
        for b in range(len(all_keypoints)):
            kp = all_keypoints[b].cpu().numpy() if isinstance(all_keypoints[b], torch.Tensor) else all_keypoints[b]
            sc = all_scores[b].cpu().numpy() if isinstance(all_scores[b], torch.Tensor) else all_scores[b]
            keypoints_np_batch.append(kp)
            scores_np_batch.append(sc)

        # Pre-process all keypoints: add neck and reorder (vectorized where possible)
        score_threshold = 0.3
        processed_keypoints = []
        processed_scores = []

        for b in range(len(keypoints_np_batch)):
            keypoints = keypoints_np_batch[b]
            scores = scores_np_batch[b]

            keypoints_with_neck = keypoints.copy()
            scores_with_neck = scores.copy()

            if len(keypoints) >= 17:
                neck = (keypoints[5] + keypoints[6]) / 2
                neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0

                keypoints_with_neck = np.insert(keypoints, 17, neck, axis=0)
                scores_with_neck = np.insert(scores, 17, neck_score)

                mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])

                temp_kpts = keypoints_with_neck.copy()
                temp_scores = scores_with_neck.copy()
                temp_kpts[openpose_idx] = keypoints_with_neck[mmpose_idx]
                temp_scores[openpose_idx] = scores_with_neck[mmpose_idx]

                keypoints_with_neck = temp_kpts
                scores_with_neck = temp_scores

            processed_keypoints.append(keypoints_with_neck)
            processed_scores.append(scores_with_neck)

        canvas_batch_np = np.zeros((len(all_keypoints), height, width, 3), dtype=np.uint8)

        # Draw all frames
        pose_outputs = []
        for b in tqdm(range(len(all_keypoints)), desc="Drawing keypoints on frames"):
            pose_canvas = canvas_batch_np[b].copy()
            drawer = KeypointDraw()
            pose_canvas = drawer.draw_wholebody_keypoints(pose_canvas, processed_keypoints[b], processed_scores[b], threshold=score_threshold,
                                                          draw_body=draw_body, draw_feet=draw_feet, draw_face=draw_face, draw_hands=draw_hands)
            pose_outputs.append(pose_canvas)

        pose_outputs_np = np.stack(pose_outputs) if len(pose_outputs) > 1 else np.expand_dims(pose_outputs[0], 0)
        final_pose_output = torch.from_numpy(pose_outputs_np).float() / 255.0

        return io.NodeOutput(final_pose_output)

class SDPoseKeypointExtractor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDPoseKeypointExtractor",
            category="image/preprocessors",
            search_aliases=["openpose", "pose detection", "preprocessor", "keypoints", "sdpose"],
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("batch_size", default=8, min=1, max=10000, step=1),
            ],
            outputs=[
                io.Custom("POSEKEYPOINTS").Output("keypoints", tooltip="Keypoints extracted from the image"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, image, batch_size) -> io.NodeOutput:

        latent_image = vae.encode(image)
        height, width = image.shape[-3], image.shape[-2]

        context = LotusConditioning().execute().result[0]

        # Use output_block_patch to capture the last 640-channel feature
        def output_patch(h, hsp, transformer_options):
            nonlocal captured_feat
            if h.shape[1] == 640:  # Capture the features for wholebody
                captured_feat = h.clone()
            return h, hsp

        model_clone = model.clone()
        model_clone.model_options["transformer_options"] = {"patches": {"output_block_patch": [output_patch]}}

        total_images = latent_image.shape[0]
        all_keypoints = []
        all_scores = []

        head = model.model.diffusion_model.heatmap_head
        pbar = comfy.utils.ProgressBar(total_images)

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            latent_batch = latent_image[batch_start:batch_end]

            captured_feat = None
            _ = comfy.sample.sample(model_clone, noise=torch.zeros_like(latent_batch), steps=1, cfg=1.0, sampler_name="euler",
                                    scheduler="simple", positive=context, negative=context, latent_image=latent_batch, disable_noise=True)

            keypoints_batch, scores_batch = head(captured_feat)
            all_keypoints.extend(keypoints_batch)
            all_scores.extend(scores_batch)

            pbar.update(batch_end - batch_start)

        out = {"keypoints": all_keypoints, "scores": all_scores, "image_size": (height, width)}
        return io.NodeOutput(out)


class SDPoseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SDPoseKeypointExtractor,
            SDPoseDrawKeypoints,
        ]

async def comfy_entrypoint() -> SDPoseExtension:
    return SDPoseExtension()
