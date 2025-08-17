import os
import threading
import traceback

from aiohttp import web

import impact
import folder_paths

import torchvision

import impact.core as core
import impact.impact_pack as impact_pack
from impact.utils import to_tensor
import impact.utils as utils
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import nodes
from PIL import Image
import io
import comfy
from io import BytesIO
import random
from server import PromptServer
import logging


sam_predictor = None
default_sam_model_name = os.path.join(impact_pack.model_path, "sams", "sam_vit_b_01ec64.pth")

sam_lock = threading.Condition()

last_prepare_data = None


def async_prepare_sam(image_dir, model_name, filename):
    with sam_lock:
        global sam_predictor

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam_model = sam_model_registry[model_kind](checkpoint=model_name)
        sam_predictor = SamPredictor(sam_model)

        image_path = os.path.join(image_dir, filename)
        image = nodes.LoadImage().load_image(image_path)[0]
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        if impact.config.get_config()['sam_editor_cpu']:
            device = 'cpu'
        else:
            device = comfy.model_management.get_torch_device()

        sam_predictor.model.to(device=device)
        sam_predictor.set_image(image, "RGB")
        sam_predictor.model.cpu()


@PromptServer.instance.routes.post("/sam/prepare")
async def sam_prepare(request):
    global sam_predictor
    global last_prepare_data
    data = await request.json()

    with sam_lock:
        if last_prepare_data is not None and last_prepare_data == data:
            # already loaded: skip -- prevent redundant loading
            return web.Response(status=200)

        last_prepare_data = data

        model_name = 'sam_vit_b_01ec64.pth'
        if data['sam_model_name'] == 'auto':
            model_name = impact.config.get_config()['sam_editor_model']

        model_path = folder_paths.get_full_path("sams", model_name)

        if model_path is None:
            logging.error(f"[Impact Pack] The '{model_name}' model file cannot be found in any sams model path.")
            return web.Response(status=400)

        logging.info(f"[Impact Pack] Loading SAM model '{model_path}'")

        filename, image_dir = folder_paths.annotated_filepath(data["filename"])

        if image_dir is None:
            typ = data['type'] if data['type'] != '' else 'output'
            image_dir = folder_paths.get_directory_by_type(typ)
            if data['subfolder'] is not None and data['subfolder'] != '':
                image_dir += f"/{data['subfolder']}"

        if image_dir is None:
            return web.Response(status=400)

        thread = threading.Thread(target=async_prepare_sam, args=(image_dir, model_path, filename,))
        thread.start()

        logging.info("[Impact Pack] SAM model loaded. ")
    return web.Response(status=200)


@PromptServer.instance.routes.post("/sam/release")
async def release_sam(request):
    global sam_predictor

    with sam_lock:
        temp = sam_predictor
        del temp
        sam_predictor = None

    logging.info("[Impact Pack]: unloading SAM model")


@PromptServer.instance.routes.post("/sam/detect")
async def sam_detect(request):
    global sam_predictor
    with sam_lock:
        if sam_predictor is not None:
            if impact.config.get_config()['sam_editor_cpu']:
                device = 'cpu'
            else:
                device = comfy.model_management.get_torch_device()

            sam_predictor.model.to(device=device)
            try:
                data = await request.json()

                positive_points = data['positive_points']
                negative_points = data['negative_points']
                threshold = data['threshold']

                points = []
                plabs = []

                for p in positive_points:
                    points.append(p)
                    plabs.append(1)

                for p in negative_points:
                    points.append(p)
                    plabs.append(0)

                detected_masks = core.sam_predict(sam_predictor, points, plabs, None, threshold)
                mask = utils.combine_masks2(detected_masks)

                if mask is None:
                    return web.Response(status=400)

                image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                i = 255. * image.cpu().numpy()

                img = Image.fromarray(np.clip(i[0], 0, 255).astype(np.uint8))

                img_buffer = io.BytesIO()
                img.save(img_buffer, format='png')

                headers = {'Content-Type': 'image/png'}
            finally:
                sam_predictor.model.to(device="cpu")

            return web.Response(body=img_buffer.getvalue(), headers=headers)

        else:
            return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/wildcards/refresh")
async def wildcards_refresh(request):
    impact.wildcards.wildcard_load()
    return web.Response(status=200)


@PromptServer.instance.routes.get("/impact/wildcards/list")
async def wildcards_list(request):
    data = {'data': impact.wildcards.get_wildcard_list()}
    return web.json_response(data)


@PromptServer.instance.routes.post("/impact/wildcards")
async def populate_wildcards(request):
    data = await request.json()
    populated = impact.wildcards.process(data['text'], data.get('seed', None))
    return web.json_response({"text": populated})


segs_picker_map = {}

@PromptServer.instance.routes.get("/impact/segs/picker/count")
async def segs_picker_count(request):
    node_id = request.rel_url.query.get('id', '')

    if node_id in segs_picker_map:
        res = len(segs_picker_map[node_id])
        return web.Response(status=200, text=str(res))

    return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/segs/picker/view")
async def segs_picker(request):
    node_id = request.rel_url.query.get('id', '')
    idx = int(request.rel_url.query.get('idx', ''))

    if node_id in segs_picker_map and idx < len(segs_picker_map[node_id]):
        img = to_tensor(segs_picker_map[node_id][idx]).permute(0, 3, 1, 2).squeeze(0)
        pil = torchvision.transforms.ToPILImage('RGB')(img)

        image_bytes = BytesIO()
        pil.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        return web.Response(status=200, body=image_bytes, content_type='image/png', headers={"Content-Disposition": f"filename={node_id}{idx}.png"})

    return web.Response(status=400)


@PromptServer.instance.routes.get("/view/validate")
async def view_validate(request):
    if "filename" in request.rel_url.query:
        filename = request.rel_url.query["filename"]
        subfolder = request.rel_url.query["subfolder"]
        filename, base_dir = folder_paths.annotated_filepath(filename)

        if filename == '' or filename[0] == '/' or '..' in filename:
            return web.Response(status=400)

        if base_dir is None:
            base_dir = folder_paths.get_input_directory()

        file = os.path.join(base_dir, subfolder, filename)

        if os.path.isfile(file):
            return web.Response(status=200)

    return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/validate/pb_id_image")
async def view_pb_id_image(request):
    if "id" in request.rel_url.query:
        pb_id = request.rel_url.query["id"]

        if pb_id not in core.preview_bridge_image_id_map:
            return web.Response(status=400)

        file = core.preview_bridge_image_id_map[pb_id]
        if os.path.isfile(file):
            return web.Response(status=200)

    return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/set/pb_id_image")
async def set_previewbridge_image(request):
    try:
        if "filename" in request.rel_url.query:
            node_id = request.rel_url.query["node_id"]
            filename = request.rel_url.query["filename"]
            path_type = request.rel_url.query["type"]
            subfolder = request.rel_url.query["subfolder"]
            filename, output_dir = folder_paths.annotated_filepath(filename)

            if filename == '' or filename[0] == '/' or '..' in filename:
                return web.Response(status=400)

            if output_dir is None:
                if path_type == 'input':
                    output_dir = folder_paths.get_input_directory()
                elif path_type == 'output':
                    output_dir = folder_paths.get_output_directory()
                else:
                    output_dir = folder_paths.get_temp_directory()

            file = os.path.join(output_dir, subfolder, filename)
            item = {
                'filename': filename,
                'type': path_type,
                'subfolder': subfolder,
            }
            pb_id = core.set_previewbridge_image(node_id, file, item)

            return web.Response(status=200, text=pb_id)
    except Exception:
        traceback.print_exc()

    return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/get/pb_id_image")
async def get_previewbridge_image(request):
    if "id" in request.rel_url.query:
        pb_id = request.rel_url.query["id"]

        if pb_id in core.preview_bridge_image_id_map:
            _, path_item = core.preview_bridge_image_id_map[pb_id]
            return web.json_response(path_item)

    return web.Response(status=400)


@PromptServer.instance.routes.get("/impact/view/pb_id_image")
async def view_previewbridge_image(request):
    if "id" in request.rel_url.query:
        pb_id = request.rel_url.query["id"]

        if pb_id in core.preview_bridge_image_id_map:
            file = core.preview_bridge_image_id_map[pb_id]

            with Image.open(file):
                filename = os.path.basename(file)
                return web.FileResponse(file, headers={"Content-Disposition": f"filename=\"{filename}\""})

    return web.Response(status=400)


def onprompt_for_switch(json_data):
    inversed_switch_info = {}
    onprompt_switch_info = {}
    onprompt_cond_branch_info = {}
    disabled_switch = set()


    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']
        if cls == 'ImpactInversedSwitch':
            # if 'sel_mode' is 'select_on_prompt'
            if 'sel_mode' in v['inputs'] and v['inputs']['sel_mode'] and 'select' in v['inputs']:
                select_input = v['inputs']['select']
                # if 'select' is converted input
                if isinstance(select_input, list) and len(select_input) == 2:
                    input_node = json_data['prompt'][select_input[0]]
                    if input_node['class_type'] == 'ImpactInt' and 'inputs' in input_node and 'value' in input_node['inputs']:
                        inversed_switch_info[k] = input_node['inputs']['value']
                    else:
                        logging.warning(f"\n##### ##### #####\n[Impact Pack] {cls}: For the 'select' operation, only 'select_index' of the 'ImpactInversedSwitch', which is not an input, or 'ImpactInt' and 'Primitive' are allowed as inputs if 'select_on_prompt' is selected.\n##### ##### #####\n")
                else:
                    inversed_switch_info[k] = select_input

        elif cls in ['ImpactSwitch', 'LatentSwitch', 'SEGSSwitch', 'ImpactMakeImageList']:
            # if 'sel_mode' is 'select_on_prompt'
            if 'sel_mode' in v['inputs'] and v['inputs']['sel_mode'] and 'select' in v['inputs']:
                select_input = v['inputs']['select']
                # if 'select' is converted input
                if isinstance(select_input, list) and len(select_input) == 2:
                    input_node = json_data['prompt'][select_input[0]]
                    if input_node['class_type'] == 'ImpactInt' and 'inputs' in input_node and 'value' in input_node['inputs']:
                        onprompt_switch_info[k] = input_node['inputs']['value']
                    if input_node['class_type'] == 'ImpactSwitch' and 'inputs' in input_node and 'select' in input_node['inputs']:
                        if isinstance(input_node['inputs']['select'], int):
                            onprompt_switch_info[k] = input_node['inputs']['select']
                        else:
                            logging.warning(f"\n##### ##### #####\n[Impact Pack] {cls}: For the 'select' operation, only 'select_index' of the 'ImpactSwitch', which is not an input, or 'ImpactInt' and 'Primitive' are allowed as inputs if 'select_on_prompt' is selected.\n##### ##### #####\n")
                else:
                    onprompt_switch_info[k] = select_input

                if k in onprompt_switch_info and f'input{onprompt_switch_info[k]}' not in v['inputs']:
                    # disconnect output
                    disabled_switch.add(k)

        elif cls == 'ImpactConditionalBranchSelMode':
            if 'sel_mode' in v['inputs'] and v['inputs']['sel_mode'] and 'cond' in v['inputs']:
                cond_input = v['inputs']['cond']
                if isinstance(cond_input, list) and len(cond_input) == 2:
                    input_node = json_data['prompt'][cond_input[0]]
                    if (input_node['class_type'] == 'ImpactValueReceiver' and 'inputs' in input_node
                            and 'value' in input_node['inputs'] and 'typ' in input_node['inputs']):
                        if 'BOOLEAN' == input_node['inputs']['typ']:
                            try:
                                onprompt_cond_branch_info[k] = input_node['inputs']['value'].lower() == "true"
                            except Exception:
                                pass
                else:
                    onprompt_cond_branch_info[k] = cond_input

    for k, v in json_data['prompt'].items():
        disable_targets = set()

        for kk, vv in v['inputs'].items():
            if isinstance(vv, list) and len(vv) == 2:
                if vv[0] in inversed_switch_info:
                    if vv[1] + 1 != inversed_switch_info[vv[0]]:
                        disable_targets.add(kk)
                    else:
                        del inversed_switch_info[k]

                if vv[0] in disabled_switch:
                    disable_targets.add(kk)

        if k in onprompt_switch_info:
            selected_slot_name = f"input{onprompt_switch_info[k]}"
            for kk, vv in v['inputs'].items():
                if kk != selected_slot_name and kk.startswith('input'):
                    disable_targets.add(kk)

        if k in onprompt_cond_branch_info:
            selected_slot_name = "tt_value" if onprompt_cond_branch_info[k] else "ff_value"
            for kk, vv in v['inputs'].items():
                if kk in ['tt_value', 'ff_value'] and kk != selected_slot_name:
                    disable_targets.add(kk)

        for kk in disable_targets:
            del v['inputs'][kk]

    # inversed_switch - select out of range
    for target in inversed_switch_info.keys():
        del json_data['prompt'][target]['inputs']['input']


def onprompt_for_pickers(json_data):
    detected_pickers = set()

    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']
        if cls == 'ImpactSEGSPicker':
            detected_pickers.add(k)

    # garbage collection
    keys_to_remove = [key for key in segs_picker_map if key not in detected_pickers]
    for key in keys_to_remove:
        del segs_picker_map[key]


def gc_preview_bridge_cache(json_data):
    prompt_keys = json_data['prompt'].keys()

    for key in list(core.preview_bridge_cache.keys()):
        if key not in prompt_keys:
            # print(f"key deleted [PB]: {key}")
            del core.preview_bridge_cache[key]

    for key in list(core.preview_bridge_last_mask_cache.keys()):
        if key not in prompt_keys:
            # print(f"key deleted [PB_last_mask]: {key}")
            del core.preview_bridge_last_mask_cache[key]


def workflow_imagereceiver_update(json_data):
    prompt = json_data['prompt']

    for v in prompt.values():
        if 'class_type' in v and v['class_type'] == 'ImageReceiver':
            if v['inputs']['save_to_workflow']:
                v['inputs']['image'] = "#DATA"


def regional_sampler_seed_update(json_data):
    prompt = json_data['prompt']

    for k, v in prompt.items():
        if 'class_type' in v and v['class_type'] == 'RegionalSampler':
            seed_2nd_mode = v['inputs']['seed_2nd_mode']

            new_seed = None
            if seed_2nd_mode == 'increment':
                new_seed = v['inputs']['seed_2nd']+1
                if new_seed > 1125899906842624:
                    new_seed = 0
            elif seed_2nd_mode == 'decrement':
                new_seed = v['inputs']['seed_2nd']-1
                if new_seed < 0:
                    new_seed = 1125899906842624
            elif seed_2nd_mode == 'randomize':
                new_seed = random.randint(0, 1125899906842624)

            if new_seed is not None:
                PromptServer.instance.send_sync("impact-node-feedback", {"node_id": k, "widget_name": "seed_2nd", "type": "INT", "value": new_seed})


def onprompt_populate_wildcards(json_data):
    prompt = json_data['prompt']

    updated_widget_values = {}
    for k, v in prompt.items():
        if 'class_type' in v and (v['class_type'] == 'ImpactWildcardEncode' or v['class_type'] == 'ImpactWildcardProcessor'):
            inputs = v['inputs']

            # legacy adapter
            if isinstance(inputs['mode'], bool):
                if inputs['mode']:
                    new_mode = 'populate'
                else:
                    new_mode = 'fixed'

                inputs['mode'] = new_mode

            if inputs['mode'] == 'populate' and isinstance(inputs['populated_text'], str):
                if isinstance(inputs['seed'], list):
                    try:
                        input_node = prompt[inputs['seed'][0]]
                        if input_node['class_type'] == 'ImpactInt':
                            input_seed = int(input_node['inputs']['value'])
                            if not isinstance(input_seed, int):
                                continue
                        if input_node['class_type'] == 'Seed (rgthree)':
                            input_seed = int(input_node['inputs']['seed'])
                            if not isinstance(input_seed, int):
                                continue
                        else:
                            logging.info(f"[Impact Pack] Only `ImpactInt`, `Seed (rgthree)` and `Primitive` Node are allowed as the seed for '{v['class_type']}'. It will be ignored. ")
                            continue
                    except Exception:
                        continue
                else:
                    input_seed = int(inputs['seed'])

                inputs['populated_text'] = impact.wildcards.process(inputs['wildcard_text'], input_seed)
                inputs['mode'] = 'reproduce'

                PromptServer.instance.send_sync("impact-node-feedback", {"node_id": k, "widget_name": "populated_text", "type": "STRING", "value": inputs['populated_text']})
                updated_widget_values[k] = inputs['populated_text']

            if inputs['mode'] == 'reproduce':
                PromptServer.instance.send_sync("impact-node-feedback", {"node_id": k, "widget_name": "mode", "type": "STRING", "value": 'populate'})



    if 'extra_data' in json_data and 'extra_pnginfo' in json_data['extra_data']:
        for node in json_data['extra_data']['extra_pnginfo']['workflow']['nodes']:
            key = str(node['id'])
            if key in updated_widget_values:
                node['widgets_values'][1] = updated_widget_values[key]
                node['widgets_values'][2] = 'reproduce'


def onprompt_for_remote(json_data):
    prompt = json_data['prompt']

    for v in prompt.values():
        if 'class_type' in v:
            cls = v['class_type']
            if cls == 'ImpactRemoteBoolean' or cls == 'ImpactRemoteInt':
                inputs = v['inputs']
                node_id = str(inputs['node_id'])

                if node_id not in prompt:
                    continue

                target_inputs = prompt[node_id]['inputs']

                widget_name = inputs['widget_name']
                if widget_name in target_inputs:
                    widget_type = None
                    if cls == 'ImpactRemoteBoolean' and isinstance(target_inputs[widget_name], bool):
                        widget_type = 'BOOLEAN'

                    elif cls == 'ImpactRemoteInt' and (isinstance(target_inputs[widget_name], int) or isinstance(target_inputs[widget_name], float)):
                        widget_type = 'INT'

                    if widget_type is None:
                        break

                    target_inputs[widget_name] = inputs['value']
                    PromptServer.instance.send_sync("impact-node-feedback", {"node_id": node_id, "widget_name": widget_name, "type": widget_type, "value": inputs['value']})


def onprompt(json_data):
    try:
        onprompt_for_remote(json_data)  # NOTE: top priority
        onprompt_for_switch(json_data)
        onprompt_for_pickers(json_data)
        onprompt_populate_wildcards(json_data)
        gc_preview_bridge_cache(json_data)
        workflow_imagereceiver_update(json_data)
        regional_sampler_seed_update(json_data)
        core.current_prompt = json_data
    except Exception as e:
        logging.warning(f"[Impact Pack] ComfyUI-Impact-Pack: Error on prompt - several features will not work.\n{e}")

    return json_data


PromptServer.instance.add_on_prompt_handler(onprompt)
