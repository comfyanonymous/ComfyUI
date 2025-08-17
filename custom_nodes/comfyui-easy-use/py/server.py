import random
import server
from enum import Enum

class SGmode(Enum):
    FIX = 1
    INCR = 2
    DECR = 3
    RAND = 4


class SeedGenerator:
    def __init__(self, base_value, action):
        self.base_value = base_value

        if action == "fixed" or action == "increment" or action == "decrement" or action == "randomize":
            self.action = SGmode.FIX
        elif action == 'increment for each node':
            self.action = SGmode.INCR
        elif action == 'decrement for each node':
            self.action = SGmode.DECR
        elif action == 'randomize for each node':
            self.action = SGmode.RAND

    def next(self):
        seed = self.base_value

        if self.action == SGmode.INCR:
            self.base_value += 1
            if self.base_value > 1125899906842624:
                self.base_value = 0
        elif self.action == SGmode.DECR:
            self.base_value -= 1
            if self.base_value < 0:
                self.base_value = 1125899906842624
        elif self.action == SGmode.RAND:
            self.base_value = random.randint(0, 1125899906842624)

        return seed


def control_seed(v, action, seed_is_global):
    action = v['inputs']['action'] if seed_is_global else action
    value = v['inputs']['value'] if seed_is_global else v['inputs']['seed_num']

    if action == 'increment' or action == 'increment for each node':
        value = value + 1
        if value > 1125899906842624:
            value = 0
    elif action == 'decrement' or action == 'decrement for each node':
        value = value - 1
        if value < 0:
            value = 1125899906842624
    elif action == 'randomize' or action == 'randomize for each node':
        value = random.randint(0, 1125899906842624)
    if seed_is_global:
        v['inputs']['value'] = value

    return value


def prompt_seed_update(json_data):
    try:
        seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']
    except:
        return None

    workflow = json_data['extra_data']['extra_pnginfo']['workflow']
    seed_widget_map = workflow['seed_widgets']
    value = None
    mode = None
    node = None
    action = None
    seed_is_global = False

    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']

        if cls == 'easy globalSeed':
            mode = v['inputs']['mode']
            action = v['inputs']['action']
            value = v['inputs']['value']
            node = k, v
            seed_is_global = True

    # control before generated
    if mode is not None and mode and seed_is_global:
        value = control_seed(node[1], action, seed_is_global)

    if seed_is_global:
        if value is not None:
            seed_generator = SeedGenerator(value, action)

            for k, v in json_data['prompt'].items():
                for k2, v2 in v['inputs'].items():
                    if isinstance(v2, str) and '$GlobalSeed.value$' in v2:
                        v['inputs'][k2] = v2.replace('$GlobalSeed.value$', str(value))

                if k not in seed_widget_map:
                    continue

                if 'seed_num' in v['inputs']:
                    if isinstance(v['inputs']['seed_num'], int):
                        v['inputs']['seed_num'] = seed_generator.next()

                if 'seed' in v['inputs']:
                    if isinstance(v['inputs']['seed'], int):
                        v['inputs']['seed'] = seed_generator.next()

                if 'noise_seed' in v['inputs']:
                    if isinstance(v['inputs']['noise_seed'], int):
                        v['inputs']['noise_seed'] = seed_generator.next()

                for k2, v2 in v['inputs'].items():
                    if isinstance(v2, str) and '$GlobalSeed.value$' in v2:
                        v['inputs'][k2] = v2.replace('$GlobalSeed.value$', str(value))
        # control after generated
        if mode is not None and not mode:
            control_seed(node[1], action, seed_is_global)

    return value is not None


def workflow_seed_update(json_data):
    nodes = json_data['extra_data']['extra_pnginfo']['workflow']['nodes']
    seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']
    prompt = json_data['prompt']

    updated_seed_map = {}
    value = None

    for node in nodes:
        node_id = str(node['id'])
        if node_id in prompt:
            if node['type'] == 'easy globalSeed':
                value = prompt[node_id]['inputs']['value']
                length = len(node['widgets_values'])
                node['widgets_values'][length-1] = node['widgets_values'][0]
                node['widgets_values'][0] = value
            elif node_id in seed_widget_map:
                widget_idx = seed_widget_map[node_id]

                if 'seed_num' in prompt[node_id]['inputs']:
                    seed = prompt[node_id]['inputs']['seed_num']
                elif 'noise_seed' in prompt[node_id]['inputs']:
                    seed = prompt[node_id]['inputs']['noise_seed']
                else:
                    seed = prompt[node_id]['inputs']['seed']

                node['widgets_values'][widget_idx] = seed
                updated_seed_map[node_id] = seed

    server.PromptServer.instance.send_sync("easyuse-global-seed", {"id": node_id, "value": value, "seed_map": updated_seed_map})


def onprompt(json_data):
    is_changed = prompt_seed_update(json_data)
    if is_changed:
        workflow_seed_update(json_data)

    return json_data

server.PromptServer.instance.add_on_prompt_handler(onprompt)