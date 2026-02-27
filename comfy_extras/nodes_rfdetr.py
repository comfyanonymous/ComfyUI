from typing_extensions import override

import torch
from comfy.ldm.rf_detr.rfdetr_v4 import COCO_CLASSES
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont


class RFDETR_detect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RFDETR_detect",
            display_name="RF-DETR Detect",
            category="detection/",
            search_aliases=["bbox", "bounding box", "object detection", "coco"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Float.Input("threshold", display_name="threshold", default=0.5),
                io.Combo.Input("class_name", options=["all"] + COCO_CLASSES, default="all", tooltip="Filter detections by class. Set to 'all' to disable filtering."),
                io.Int.Input("max_detections", display_name="max_detections", default=100, tooltip="Maximum number of detections to return per image. In order of descending confidence score."),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes")],
        )

    @classmethod
    def execute(cls, model, image, threshold, class_name, max_detections) -> io.NodeOutput:
        B, H, W, C = image.shape

        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 640, 640, "bilinear", crop="disabled")

        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype_inference()
        orig_size = torch.tensor([[W, H]], device=device, dtype=dtype).expand(B, -1)  # [B, 2] as (W, H)

        comfy.model_management.load_model_gpu(model)
        results = model.model.diffusion_model(image_in.to(device=device, dtype=dtype), orig_size)  # list of B dicts

        all_bbox_dicts = []

        def _postprocess(results, threshold=0.5):
            det   = results[0]
            keep  = det['scores'] > threshold
            return det['boxes'][keep].cpu(), det['labels'][keep].cpu(), det['scores'][keep].cpu()

        for i in range(B):
            boxes, labels, scores = _postprocess(results[i:i+1], threshold=threshold)

            bbox_dicts = sorted([
                {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                    "label": COCO_CLASSES[int(label)],
                    "score": float(score)
                }
                for box, label, score in zip(boxes, labels, scores)
                if class_name == "all" or COCO_CLASSES[int(label)] == class_name
            ], key=lambda d: d["score"], reverse=True)[:max_detections]
            all_bbox_dicts.append(bbox_dicts)

        return io.NodeOutput(all_bbox_dicts)


class DrawBBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DrawBBoxes",
            display_name="Draw BBoxes",
            category="detection/",
            search_aliases=["bbox", "bounding box", "object detection", "rf_detr", "visualize detections", "coco"],
            inputs=[
                io.Image.Input("image", optional=True),
                io.BoundingBox.Input("bboxes", force_input=True),
            ],
            outputs=[
                io.Image.Output("out_image"),
            ],
        )

    @classmethod
    def execute(cls, bboxes, image=None) -> io.NodeOutput:
        # Normalise bboxes to a list-of-lists (one list of detections per image).
        # It may arrive as: a bare dict, a flat list of dicts, or a list of lists.
        B = image.shape[0] if image is not None else 1
        if isinstance(bboxes, dict):
            bboxes = [[bboxes]] * B
        elif not isinstance(bboxes, list) or len(bboxes) == 0:
            bboxes = [[]] * B
        elif not isinstance(bboxes[0], list):
            # flat list of dicts: same detections for every image
            bboxes = [bboxes] * B

        if image is None:
            B = len(bboxes)
            max_w = max((int(d["x"] + d["width"])  for frame in bboxes for d in frame), default=640)
            max_h = max((int(d["y"] + d["height"]) for frame in bboxes for d in frame), default=640)
            image = torch.zeros((B, max_h, max_w, 3), dtype=torch.float32)

        all_out_images = []
        for i in range(B):
            detections = bboxes[i]
            if detections:
                boxes  = torch.tensor([[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in detections])
                labels = [d.get("label") if d.get("label") in COCO_CLASSES else None for d in detections]
                scores = torch.tensor([d.get("score", 1.0) for d in detections])
            else:
                boxes  = torch.zeros((0, 4))
                labels = []
                scores = torch.zeros((0,))

            pil_image = image[i].movedim(-1, 0)
            img = ToPILImage()(pil_image)
            if detections:
                img = cls.draw_detections(img, boxes, labels, scores)
            all_out_images.append(ToTensor()(img).unsqueeze(0).movedim(1, -1))

        out_images = torch.cat(all_out_images, dim=0).to(comfy.model_management.intermediate_device())
        return io.NodeOutput(out_images)

    @classmethod
    def draw_detections(cls, img, boxes, labels, scores):
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except Exception:
            font = ImageFont.load_default()
        colors = [(255,0,0),(0,200,0),(0,0,255),(255,165,0),(128,0,128),
                (0,255,255),(255,20,147),(100,149,237)]
        for box, label, score in sorted(zip(boxes, labels, scores), key=lambda x: x[2].item()):
            x1, y1, x2, y2 = box.tolist()
            color_idx = COCO_CLASSES.index(label) if label is not None else 0
            c = colors[color_idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            if label is not None:
                draw.text((x1 + 2, y1 + 2), f'{label} {score:.2f}', fill=c, font=font)
        return img


class RFDETRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RFDETR_detect,
            DrawBBoxes,
        ]


async def comfy_entrypoint() -> RFDETRExtension:
    return RFDETRExtension()
