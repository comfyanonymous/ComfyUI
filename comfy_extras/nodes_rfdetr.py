from typing_extensions import override

import torch
from comfy.ldm.rf_detr.rfdetr_v4 import COCO_CLASSES
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from torchvision.transforms import ToPILImage, ToTensor


class RFDETR_detect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RFDETR_detect",
            display_name="RF-DETR Detect",
            category="detection/",
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Float.Input("threshold", display_name="threshold", default=0.5),
                io.Combo.Input("class_name", options=["all"] + COCO_CLASSES, default="all")
            ],
            outputs=[
                io.BoundingBox.Output("bbox")],
        )

    @classmethod
    def execute(cls, model, image, threshold, class_name) -> io.NodeOutput:
        B, H, W, C = image.shape

        device = comfy.model_management.get_torch_device()
        orig_size = torch.tensor([[W, H]], device=device, dtype=torch.float32).expand(B, -1)  # [B, 2] as (W, H)

        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 640, 640, "bilinear", crop="disabled")

        comfy.model_management.load_model_gpu(model)
        out     = model.model.diffusion_model(image_in.to(device=device)) # [B, num_queries, 4+num_classes]
        results = model.model.diffusion_model.postprocess(out, orig_size)  # list of B dicts

        all_bbox_dicts = []

        def _postprocess(results, threshold=0.5):
            det   = results[0]
            keep  = det['scores'] > threshold
            return det['boxes'][keep].cpu(), det['labels'][keep].cpu(), det['scores'][keep].cpu()

        for i in range(B):
            boxes, labels, scores = _postprocess(results[i:i+1], threshold=threshold)

            print(f'\nImage {i + 1}/{B}: Detected {len(boxes)} objects (threshold={threshold}):')
            for box, label, score in sorted(zip(boxes, labels, scores), key=lambda x: -x[2].item()):
                print(f'  {COCO_CLASSES[label.item()]:20s}  {score:.3f}  '
                    f'[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]')

            bbox_dicts = [
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
            ]
            all_bbox_dicts.append(bbox_dicts)

        return io.NodeOutput(all_bbox_dicts)


class RFDETR_draw(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RFDETR_draw",
            display_name="RF-DETR Draw Detections",
            category="detection/",
            inputs=[
                io.Image.Input("image", display_name="image", optional=True),
                io.BoundingBox.Input("bbox", display_name="bbox", force_input=True),
            ],
            outputs=[
                io.Image.Output("out_image", display_name="out_image"),
            ],
        )

    @classmethod
    def execute(cls, bbox, image=None) -> io.NodeOutput:
        # Normalise bbox to a list-of-lists (one list of detections per image).
        # It may arrive as: a bare dict, a flat list of dicts, or a list of lists.
        B = image.shape[0] if image is not None else 1
        if isinstance(bbox, dict):
            bbox = [[bbox]] * B
        elif not isinstance(bbox, list) or len(bbox) == 0:
            bbox = [[]] * B
        elif not isinstance(bbox[0], list):
            # flat list of dicts → same detections for every image
            bbox = [bbox] * B

        if image is None:
            image = torch.zeros((B, 3, 640, 640), dtype=torch.uint8)

        all_out_images = []
        for i in range(B):
            detections = bbox[i]
            if detections:
                boxes  = torch.tensor([[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in detections])
                labels = torch.tensor([COCO_CLASSES.index(lbl) if (lbl := d.get("label")) in COCO_CLASSES else 0 for d in detections])
                scores = torch.tensor([d.get("score", 1.0) for d in detections])
            else:
                boxes  = torch.zeros((0, 4))
                labels = torch.zeros((0,), dtype=torch.long)
                scores = torch.zeros((0,))

            pil_image = image[i].movedim(-1, 0)
            img = ToPILImage()(pil_image)
            out_image_pil = cls.draw_detections(img, boxes, labels, scores)
            all_out_images.append(ToTensor()(out_image_pil).unsqueeze(0).movedim(1, -1))

        out_images = torch.cat(all_out_images, dim=0)
        return io.NodeOutput(out_images)

    @classmethod
    def draw_detections(cls, img, boxes, labels, scores):
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except Exception:
            font = ImageFont.load_default()
        colors = [(255,0,0),(0,200,0),(0,0,255),(255,165,0),(128,0,128),
                (0,255,255),(255,20,147),(100,149,237)]
        for box, label, score in sorted(zip(boxes, labels, scores), key=lambda x: x[2].item()):
            x1, y1, x2, y2 = box.tolist()
            c = colors[label.item() % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            draw.text((x1 + 2, y1 + 2),
                    f'{COCO_CLASSES[label.item()]} {score:.2f}', fill=c, font=font)
        return img


class RFDETRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RFDETR_detect,
            RFDETR_draw,
        ]


async def comfy_entrypoint() -> RFDETRExtension:
    return RFDETRExtension()
