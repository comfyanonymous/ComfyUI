import nodes
import node_helpers
import torch
import torchvision.transforms.functional as TF
import comfy.model_management
import comfy.utils
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_wan import parse_json_tracks

# https://github.com/ali-vilab/Wan-Move/blob/main/wan/modules/trajectory.py
from PIL import Image, ImageDraw

SKIP_ZERO = False

def get_pos_emb(
    pos_k: torch.Tensor, # A 1D tensor containing positions for which to generate embeddings.
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))), #Function to compute thetas based on position and embedding dimensions.
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor: # The position embeddings (batch_size, pos_emb_dim)

    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device, dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
    # Expand denominator to match the shape needed for broadcasting
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)

    thetas = theta_func(denominator_expanded, pos_emb_dim)

    # Ensure pos_k is in the correct shape for broadcasting
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

    # Concatenate sine and cosine embeddings along the last dimension
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb

def create_pos_embeddings(
    pred_tracks: torch.Tensor, # the predicted tracks, [T, N, 2]
    pred_visibility: torch.Tensor, # the predicted visibility [T, N]
    downsample_ratios: list[int], # the ratios for downsampling time, height, and width
    height: int, # the height of the feature map
    width: int, # the width of the feature map
    track_num: int = -1, # the number of tracks to use
    t_down_strategy: str = "sample", # the strategy for downsampling time dimension
):
    assert t_down_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."

    t, n, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    track_pos = - torch.ones(n, (t-1) // t_down + 1, 2, dtype=torch.long)

    if track_num == -1:
        track_num = n

    tracks_idx = torch.randperm(n)[:track_num]
    tracks = pred_tracks[:, tracks_idx]
    visibility = pred_visibility[:, tracks_idx]

    for t_idx in range(0, t, t_down):
        if t_down_strategy == "sample" or t_idx == 0:
            cur_tracks = tracks[t_idx] # [N, 2]
            cur_visibility = visibility[t_idx] # [N]
        else:
            cur_tracks = tracks[t_idx:t_idx+t_down].mean(dim=0)
            cur_visibility = torch.any(visibility[t_idx:t_idx+t_down], dim=0)

        for i in range(track_num):
            if not cur_visibility[i] or cur_tracks[i][0] < 0 or cur_tracks[i][1] < 0 or cur_tracks[i][0] >= width or cur_tracks[i][1] >= height:
                continue
            x, y = cur_tracks[i]
            x, y = int(x // w_down), int(y // h_down)
            track_pos[i, t_idx // t_down, 0], track_pos[i, t_idx // t_down, 1] = y, x

    return track_pos # the position embeddings, [N, T', 2], 2 = height, width

def replace_feature(
    vae_feature: torch.Tensor,  # [B, C', T', H', W']
    track_pos: torch.Tensor,    # [B, N, T', 2]
) -> torch.Tensor:
    b, _, t, h, w = vae_feature.shape
    assert b == track_pos.shape[0], "Batch size mismatch."
    n = track_pos.shape[1]

    # Shuffle the trajectory order
    track_pos = track_pos[:, torch.randperm(n), :, :]

    # Extract coordinates at time steps ≥ 1 and generate a valid mask
    current_pos = track_pos[:, :, 1:, :]  # [B, N, T-1, 2]
    mask = (current_pos[..., 0] >= 0) & (current_pos[..., 1] >= 0)  # [B, N, T-1]

    # Get all valid indices
    valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 3]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return vae_feature

    # Decompose valid indices into each dimension
    batch_idx = valid_indices[:, 0]
    track_idx = valid_indices[:, 1]
    t_rel = valid_indices[:, 2]
    t_target = t_rel + 1  # Convert to original time step indices

    # Extract target position coordinates
    h_target = current_pos[batch_idx, track_idx, t_rel, 0].long()  # Ensure integer indices
    w_target = current_pos[batch_idx, track_idx, t_rel, 1].long()

    # Extract source position coordinates (t=0)
    h_source = track_pos[batch_idx, track_idx, 0, 0].long()
    w_source = track_pos[batch_idx, track_idx, 0, 1].long()

    # Get source features and assign to target positions
    src_features = vae_feature[batch_idx, :, 0, h_source, w_source]
    vae_feature[batch_idx, :, t_target, h_target, w_target] = src_features

    return vae_feature

# Visualize functions

def draw_overall_gradient_polyline_on_image(image, line_width, points, start_color, opacity=1.0):
    def get_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    new_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(new_image, 'RGBA')
    points = points[::-1]

    total_length = sum(get_distance(points[i], points[i+1]) for i in range(len(points)-1))
    accumulated_length = 0

    # Draw the gradient polyline
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = get_distance(start_point, end_point)
        steps = int(segment_length)

        for i in range(steps):
            current_length = accumulated_length + (i / steps) * segment_length

            # Alpha from fully opaque to fully transparent
            alpha = int(255 * (1 - current_length / total_length) * opacity)
            color = (*start_color, alpha)

            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            # Dynamic line width, decreasing from initial width to 1
            dynamic_line_width = int(line_width * (1 - (current_length / total_length)))
            dynamic_line_width = max(dynamic_line_width, 1)  # minimum width is 1 to avoid 0

            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length

    return new_image

def add_weighted(rgb, track):
    rgb = np.array(rgb) # [H, W, C] "RGB"
    track = np.array(track) # [H, W, C] "RGBA"

    alpha = track[:, :, 3] / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)

    return Image.fromarray(blend_img.astype(np.uint8))

def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24, circle_size=12, opacity=0.5, line_width=16):
    color_map = [(102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204), (0, 255, 0)]

    video = video.byte().cpu().numpy() # (81, 480, 832, 3)
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()

    output_frames = []
    for t in range(video.shape[0]):
        frame = video[t]
        frame = Image.fromarray(frame).convert("RGB")

        for n in range(tracks.shape[1]):
            if visibility is not None and visibility[t, n] == 0:
                continue

            # Track coordinate at current frame
            track_coord = tracks[t, n]
            tracks_coord = tracks[max(t-track_frame, 0):t+1, n]

            # Draw a circle
            overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            circle_color = color_map[n % len(color_map)] + (int(255 * opacity),)
            draw_overlay.ellipse((track_coord[0] - circle_size, track_coord[1] - circle_size, track_coord[0] + circle_size, track_coord[1] + circle_size),
                fill=circle_color
            )
            frame = add_weighted(frame, overlay)  # <-- Blend the circle overlay first
            # Draw the polyline
            track_image = draw_overall_gradient_polyline_on_image(frame, line_width, tracks_coord, color_map[n % len(color_map)], opacity=opacity)
            frame = add_weighted(frame, track_image)

        output_frames.append(frame.convert("RGB"))

    return output_frames


class WanMoveVisualizeTracks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveVisualizeTracks",
            category="conditioning/video_models",
            inputs=[
                io.Image.Input("images"),
                io.Tracks.Input("tracks", optional=True),
                io.Int.Input("line_resolution", default=24, min=1, max=1024),
                io.Int.Input("circle_size", default=12, min=1, max=128),
                io.Float.Input("opacity", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Int.Input("line_width", default=16, min=1, max=128),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, images, line_resolution, circle_size, opacity, line_width, tracks=None) -> io.NodeOutput:
        if tracks is None:
            return io.NodeOutput(images)

        track_path = tracks["track_path"].unsqueeze(0)
        track_visibility = tracks["track_visibility"].unsqueeze(0)
        images_in = images.repeat(tracks["track_path"].shape[0], 1, 1, 1) * 255.0
        track_video = draw_tracks_on_video(images_in, track_path, track_visibility, track_frame=line_resolution, circle_size=circle_size, opacity=opacity, line_width=line_width)
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).movedim(1, -1).float()

        return io.NodeOutput(track_video.to(comfy.model_management.intermediate_device()))


class WanMoveTracksFromCoords(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveTracksFromCoords",
            category="conditioning/video_models",
            inputs=[
                io.String.Input("track_coords", force_input=True, default="[]", optional=True),
                io.Mask.Input("track_mask", optional=True),
            ],
            outputs=[
                io.Tracks.Output(),
            ],
        )

    @classmethod
    def execute(cls, track_coords, track_mask=None) -> io.NodeOutput:
        device=comfy.model_management.intermediate_device()

        tracks_data = parse_json_tracks(track_coords)
        track_length = len(tracks_data[0])

        track_list = [
                [[track[frame]['x'], track[frame]['y']] for track in tracks_data]
                for frame in range(len(tracks_data[0]))
            ]
        tracks = torch.tensor(track_list, dtype=torch.float32, device=device)  # [frames, num_tracks, 2]

        num_tracks = tracks.shape[-2]
        if track_mask is None:
            track_visibility = torch.ones((track_length, num_tracks), dtype=torch.bool, device=device)
        else:
            track_visibility = (track_mask > 0).any(dim=(1, 2)).unsqueeze(-1)

        out_track_info = {}
        out_track_info["track_path"] = tracks
        out_track_info["track_visibility"] = track_visibility
        return io.NodeOutput(out_track_info)


class WanMoveConcatTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveConcatTrack",
            category="conditioning/video_models",
            inputs=[
                io.Tracks.Input("tracks_1"),
                io.Tracks.Input("tracks_2", optional=True),
            ],
            outputs=[
                io.Tracks.Output(),
            ],
        )

    @classmethod
    def execute(cls, tracks_1=None, tracks_2=None) -> io.NodeOutput:
        if tracks_2 is None:
            return io.NodeOutput(tracks_1)

        tracks_out = torch.cat([tracks_1["track_path"], tracks_2["track_path"]], dim=1)  # Concatenate along the track dimension
        mask_out = torch.cat([tracks_1["track_visibility"], tracks_2["track_visibility"]], dim=-1)

        out_track_info = {}
        out_track_info["track_path"] = tracks_out
        out_track_info["track_visibility"] = mask_out
        return io.NodeOutput(out_track_info)


class WanMoveTrackToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveTrackToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Tracks.Input("tracks", optional=True),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image"),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, tracks=None, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        device=comfy.model_management.intermediate_device()
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=device)
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if tracks is not None:
                tracks_path = tracks["track_path"][:length]  # [T, N, 2]
                num_tracks = tracks_path.shape[-2]

                track_visibility = tracks.get("track_visibility", torch.ones((length, num_tracks), dtype=torch.bool, device=device))

                track_pos = create_pos_embeddings(tracks_path, track_visibility, [4, 8, 8], height, width, track_num=num_tracks)
                track_pos = comfy.utils.resize_to_batch_size(track_pos.unsqueeze(0), batch_size)
                concat_latent_image_pos = replace_feature(concat_latent_image, track_pos)
            else:
                concat_latent_image_pos = concat_latent_image

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_pos, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class WanMoveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanMoveTrackToVideo,
            WanMoveTracksFromCoords,
            WanMoveConcatTrack,
            WanMoveVisualizeTracks,
        ]

async def comfy_entrypoint() -> WanMoveExtension:
    return WanMoveExtension()
