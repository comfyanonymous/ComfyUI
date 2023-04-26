import os
import sys
import copy
import json
import threading
import heapq
import traceback
import gc

import torch
import nodes

import comfy.model_management

def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, extra_data={}):
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                return None
            obj = outputs[input_unique_id][output_index]
            input_data_all[x] = obj
        else:
            if ("required" in valid_inputs and x in valid_inputs["required"]) or ("optional" in valid_inputs and x in valid_inputs["optional"]):
                input_data_all[x] = input_data

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = prompt
            if h[x] == "EXTRA_PNGINFO":
                if "extra_pnginfo" in extra_data:
                    input_data_all[x] = extra_data['extra_pnginfo']
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = unique_id
    return input_data_all

def recursive_execute(server, prompt, outputs, current_item, extra_data, executed):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        return

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                recursive_execute(server, prompt, outputs, input_unique_id, extra_data, executed)

    input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)
    if server.client_id is not None:
        server.last_node_id = unique_id
        server.send_sync("executing", { "node": unique_id }, server.client_id)
    obj = class_def()

    nodes.before_node_execution()
    outputs[unique_id] = getattr(obj, obj.FUNCTION)(**input_data_all)
    if "ui" in outputs[unique_id]:
        if server.client_id is not None:
            server.send_sync("executed", { "node": unique_id, "output": outputs[unique_id]["ui"] }, server.client_id)
        if "result" in outputs[unique_id]:
            outputs[unique_id] = outputs[unique_id]["result"]
    executed.add(unique_id)

def recursive_will_execute(prompt, outputs, current_item):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    will_execute = []
    if unique_id in outputs:
        return []

    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                will_execute += recursive_will_execute(prompt, outputs, input_unique_id)

    return will_execute + [unique_id]

def recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    is_changed_old = ''
    is_changed = ''
    if hasattr(class_def, 'IS_CHANGED'):
        if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
            is_changed_old = old_prompt[unique_id]['is_changed']
        if 'is_changed' not in prompt[unique_id]:
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs)
            if input_data_all is not None:
                is_changed = class_def.IS_CHANGED(**input_data_all)
                prompt[unique_id]['is_changed'] = is_changed
        else:
            is_changed = prompt[unique_id]['is_changed']

    if unique_id not in outputs:
        return True

    to_delete = False
    if is_changed != is_changed_old:
        to_delete = True
    elif unique_id not in old_prompt:
        to_delete = True
    elif inputs == old_prompt[unique_id]['inputs']:
        for x in inputs:
            input_data = inputs[x]

            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id in outputs:
                    to_delete = recursive_output_delete_if_changed(prompt, old_prompt, outputs, input_unique_id)
                else:
                    to_delete = True
                if to_delete:
                    break
    else:
        to_delete = True

    if to_delete:
        d = outputs.pop(unique_id)
        del d
    return to_delete

class PromptExecutor:
    def __init__(self, server):
        self.outputs = {}
        self.old_prompt = {}
        self.server = server

    def execute(self, prompt, extra_data={}):
        nodes.interrupt_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        with torch.inference_mode():
            #delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d

            for x in prompt:
                recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)

            current_outputs = set(self.outputs.keys())
            executed = set()
            try:
                to_execute = []
                for x in prompt:
                    class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
                    if hasattr(class_, 'OUTPUT_NODE'):
                        to_execute += [(0, x)]

                while len(to_execute) > 0:
                    #always execute the output that depends on the least amount of unexecuted nodes first
                    to_execute = sorted(list(map(lambda a: (len(recursive_will_execute(prompt, self.outputs, a[-1])), a[-1]), to_execute)))
                    x = to_execute.pop(0)[-1]

                    class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
                    if hasattr(class_, 'OUTPUT_NODE'):
                        if class_.OUTPUT_NODE == True:
                            valid = False
                            try:
                                m = validate_inputs(prompt, x)
                                valid = m[0]
                            except:
                                valid = False
                            if valid:
                                recursive_execute(self.server, prompt, self.outputs, x, extra_data, executed)
            except Exception as e:
                print(traceback.format_exc())
                to_delete = []
                for o in self.outputs:
                    if (o not in current_outputs) and (o not in executed):
                        to_delete += [o]
                        if o in self.old_prompt:
                            d = self.old_prompt.pop(o)
                            del d
                for o in to_delete:
                    d = self.outputs.pop(o)
                    del d
            finally:
                for x in executed:
                    self.old_prompt[x] = copy.deepcopy(prompt[x])
                self.server.last_node_id = None
                if self.server.client_id is not None:
                    self.server.send_sync("executing", { "node": None }, self.server.client_id)

        gc.collect()
        comfy.model_management.soft_empty_cache()


def validate_inputs(prompt, item):
    unique_id = item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    required_inputs = class_inputs['required']
    for x in required_inputs:
        if x not in inputs:
            return (False, "Required input is missing. {}, {}".format(class_type, x))
        val = inputs[x]
        info = required_inputs[x]
        type_input = info[0]
        if isinstance(val, list):
            if len(val) != 2:
                return (False, "Bad Input. {}, {}".format(class_type, x))
            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            if r[val[1]] != type_input:
                return (False, "Return type mismatch. {}, {}, {} != {}".format(class_type, x, r[val[1]], type_input))
            r = validate_inputs(prompt, o_id)
            if r[0] == False:
                return r
        else:
            if type_input == "INT":
                val = int(val)
                inputs[x] = val
            if type_input == "FLOAT":
                val = float(val)
                inputs[x] = val
            if type_input == "STRING":
                val = str(val)
                inputs[x] = val

            if len(info) > 1:
                if "min" in info[1] and val < info[1]["min"]:
                    return (False, "Value smaller than min. {}, {}".format(class_type, x))
                if "max" in info[1] and val > info[1]["max"]:
                    return (False, "Value bigger than max. {}, {}".format(class_type, x))

            if hasattr(obj_class, "VALIDATE_INPUTS"):
                input_data_all = get_input_data(inputs, obj_class, unique_id)
                ret = obj_class.VALIDATE_INPUTS(**input_data_all)
                if ret != True:
                    return (False, "{}, {}".format(class_type, ret))
            else:
                if isinstance(type_input, list):
                    if val not in type_input:
                        return (False, "Value not in list. {}, {}: {} not in {}".format(class_type, x, val, type_input))
    return (True, "")

def validate_prompt(prompt):
    outputs = set()
    for x in prompt:
        class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE == True:
            outputs.add(x)

    if len(outputs) == 0:
        return (False, "Prompt has no outputs")

    good_outputs = set()
    errors = []
    for o in outputs:
        valid = False
        reason = ""
        try:
            m = validate_inputs(prompt, o)
            valid = m[0]
            reason = m[1]
        except Exception as e:
            print(traceback.format_exc())
            valid = False
            reason = "Parsing error"

        if valid == True:
            good_outputs.add(x)
        else:
            print("Failed to validate prompt for output {} {}".format(o, reason))
            print("output will be ignored")
            errors += [(o, reason)]

    if len(good_outputs) == 0:
        errors_list = "\n".join(set(map(lambda a: "{}".format(a[1]), errors)))
        return (False, "Prompt has no properly connected outputs\n {}".format(errors_list))

    return (True, "")


class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        server.prompt_queue = self

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    def task_done(self, item_id, outputs):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            self.history[prompt[1]] = { "prompt": prompt, "outputs": {} }
            for o in outputs:
                if "ui" in outputs[o]:
                    self.history[prompt[1]]["outputs"][o] = outputs[o]["ui"]
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self):
        with self.mutex:
            return copy.deepcopy(self.history)

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)
