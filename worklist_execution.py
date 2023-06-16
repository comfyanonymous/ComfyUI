# This module modifies the original code for experimentation with a new execution structure intended to replace the original execution structure.
# Original: https://github.com/comfyanonymous/ComfyUI/blob/master/execution.py

import sys
import copy
import traceback
import gc
import time

import torch
import nodes

import comfy.model_management
from execution import get_input_data, get_output_data, map_node_over_list, format_value, full_type_name
from queue import Queue

DEBUG_FLAG = True


def print_dbg(x):
    if DEBUG_FLAG:
        print(f"[DBG] {x()}")


# To handling virtual node
class DummyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "doit"

    def doit(s, *args, **kwargs):
        if len(kwargs) == 1:
            key = list(kwargs.keys())[0]
            output = kwargs[key]
            return (output,)
        else:
            pass


def exception_helper(unique_id, input_data_all, executed, outputs, task):
    try:
        task()
        return None
    except comfy.model_management.InterruptProcessingException as iex:
        print("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": unique_id,
        }

        return executed, False, error_details, iex
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in outputs.items():
            output_data_formatted[node_id] = [[format_value(x) for x in l] for l in node_outputs]

        print("!!! Exception during processing !!!")
        print(traceback.format_exc())

        error_details = {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted
        }
        return executed, False, error_details, ex


def is_incomplete_input_slots(class_def, inputs, outputs):
    required_inputs = set(class_def.INPUT_TYPES().get("required", []))

    if len(required_inputs - inputs.keys()) > 0:
        return True

    if class_def.__name__ == "LoopControl":
        inputs = {
                    'loop_condition': inputs['loop_condition'],
                    'initial_input': inputs['initial_input'],
                  }

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            if input_unique_id not in outputs:
                return True

    return False


def get_class_def(prompt, unique_id):
    if 'class_type' not in prompt[unique_id]:
        class_def = DummyNode
    else:
        class_type = prompt[unique_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    return class_def


def get_next_nodes_map(prompt):
    next_nodes = {}
    for key, value in prompt.items():
        inputs = value['inputs']

        for input_data in inputs.values():
            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                if input_unique_id in next_nodes:
                    next_nodes[input_unique_id].add(key)
                else:
                    next_nodes[input_unique_id] = {key}
    return next_nodes


def worklist_execute(server, prompt, outputs, extra_data, prompt_id, outputs_ui, to_execute, next_nodes):
    worklist = Queue()
    executed = set()
    will_execute = {}

    def add_work_high(item):
        worklist.put(item)
        cnt = will_execute.get(item, 0)
        will_execute[item] = cnt + 1

    def add_work(item):
        worklist.put(item)
        cnt = will_execute.get(item, 0)
        will_execute[item] = cnt + 1

    def get_work():
        item = worklist.get()
        cnt = will_execute.get(item, 0)
        if cnt <= 0:
            del will_execute[item]
        else:
            will_execute[item] = cnt - 1

        return str(item)

    def apply_priority(items):
        high_priority = []
        low_priority = []

        for cur_id, cur_class_def in items:
            if cur_class_def.__name__ == "LoopControl":
                low_priority.append(cur_id)
            elif cur_class_def.RETURN_TYPES == ():
                high_priority.append(cur_id)
            else:
                low_priority.append(cur_id)

        return (high_priority, low_priority)

    def get_progress():
        total = len(executed)+len(will_execute.keys())
        return len(executed)/total

    # init seeds: the nodes that have their output not erased in the input slot are the seeds.
    for unique_id in to_execute:
        inputs = prompt[unique_id]['inputs']
        class_def = get_class_def(prompt, unique_id)

        if unique_id in outputs:
            continue

        if is_incomplete_input_slots(class_def, inputs, outputs):
            continue

        input_data_all = None

        def task():
            nonlocal input_data_all
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)

            if input_data_all is None:
                return

            if not is_incomplete_input_slots(class_def, prompt[unique_id]['inputs'], outputs):
                add_work(unique_id)  # add to seed if all input is properly provided

        result = exception_helper(unique_id, input_data_all, executed, outputs, task)
        if result is not None:
            return result  # error state

    while not worklist.empty():
        unique_id = get_work()

        inputs = prompt[unique_id]['inputs']
        class_def = get_class_def(prompt, unique_id)

        print_dbg(lambda: f"work: {unique_id} ({class_def.__name__}) / worklist: {list(worklist.queue)}")

        input_data_all = None

        def task():
            nonlocal input_data_all
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)

            if server.client_id is not None:
                server.last_node_id = unique_id
                server.send_sync("executing", {"node": unique_id, "prompt_id": prompt_id, "progress": get_progress()},
                                 server.client_id)
            obj = class_def()

            output_data, output_ui = get_output_data(obj, input_data_all)

            outputs[unique_id] = output_data
            if len(output_ui) > 0:
                outputs_ui[unique_id] = output_ui
                if server.client_id is not None:
                    server.send_sync("executed", {"node": unique_id, "output": output_ui, "prompt_id": prompt_id},
                                     server.client_id)
            executed.add(unique_id)

        result = exception_helper(unique_id, input_data_all, executed, outputs, task)
        if result is not None:
            return result  # error state
        else:
            if unique_id in next_nodes:
                if class_def.__name__ == "LoopControl" and outputs[unique_id] == [[None]]:
                    continue

                candidates = []
                for next_node in next_nodes[unique_id]:
                    if next_node in to_execute:
                        # If all input slots are not completed, do not add to the work.
                        # This prevents duplicate entries of the same work in the worklist.
                        # For loop support, it is important to fire only once when the input slot is completed.
                        next_class_def = get_class_def(prompt, next_node)
                        if not is_incomplete_input_slots(next_class_def, prompt[next_node]['inputs'], outputs):
                            candidates.append((next_node, next_class_def))

                high_priority_works, low_priority_works = apply_priority(candidates)

                for next_node in high_priority_works:
                    add_work_high(next_node)

                for next_node in low_priority_works:
                    add_work(next_node)

    return executed, True, None, None


def worklist_will_execute(prompt, outputs, worklist):
    visited = set()

    will_execute = []

    while worklist:
        unique_id = str(worklist.pop())

        if unique_id in visited:  # to avoid infinite loop and redundant processing
            continue
        else:
            visited.add(unique_id)

        inputs = prompt[unique_id]['inputs']

        if unique_id in outputs:
            continue

        for x in inputs:
            input_data = inputs[x]
            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id not in outputs:
                    worklist.append(input_unique_id)

        will_execute.append(unique_id)

    return will_execute


def worklist_output_delete_if_changed(prompt, old_prompt, outputs, next_nodes):
    worklist = []
    deleted = set()

    # init seeds
    for unique_id, value in prompt.items():
        inputs = value['inputs']
        if 'class_type' not in value:
            class_def = DummyNode
        else:
            class_type = value['class_type']
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

        is_changed_old = ''
        is_changed = ''
        to_delete = False

        if hasattr(class_def, 'IS_CHANGED'):
            if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
                is_changed_old = old_prompt[unique_id]['is_changed']
            if 'is_changed' not in value:
                input_data_all = get_input_data(inputs, class_def, unique_id, outputs)
                if input_data_all is not None:
                    try:
                        is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
                        value['is_changed'] = is_changed
                    except:
                        to_delete = True
            else:
                is_changed = value['is_changed']

        if unique_id not in outputs:
            to_delete = True
        else:
            if not to_delete:
                if is_changed != is_changed_old:
                    to_delete = True
                elif unique_id not in old_prompt:
                    to_delete = True
                elif inputs == old_prompt[unique_id]['inputs']:
                    for x in inputs:
                        input_data = inputs[x]

                        if isinstance(input_data, list):
                            input_unique_id = input_data[0]

                            if input_unique_id not in outputs:
                                to_delete = True
                                break
                else:
                    to_delete = True

        if to_delete:
            worklist.append(unique_id)

    # cascade removing
    while worklist:
        unique_id = worklist.pop()

        if unique_id in deleted:
            continue

        if unique_id in outputs:
            d = outputs.pop(unique_id)
            del d

        new_works = next_nodes.get(unique_id, [])
        worklist.extend(new_works)
        deleted.add(unique_id)

    return outputs


class PromptExecutor:
    def __init__(self, server):
        self.outputs = {}
        self.outputs_ui = {}
        self.old_prompt = {}
        self.server = server

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        if "class_type" in prompt[node_id]:
            class_type = prompt[node_id]["class_type"]
        else:
            class_type = "ComponentInput/Output"

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.server.send_sync("execution_interrupted", mes, self.server.client_id)
        else:
            if self.server.client_id is not None:
                mes = {
                    "prompt_id": prompt_id,
                    "node_id": node_id,
                    "node_type": class_type,
                    "executed": list(executed),

                    "exception_message": error["exception_message"],
                    "exception_type": error["exception_type"],
                    "traceback": error["traceback"],
                    "current_inputs": error["current_inputs"],
                    "current_outputs": error["current_outputs"],
                }
                self.server.send_sync("execution_error", mes, self.server.client_id)

        # Next, remove the subsequent outputs since they will not be executed
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

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        execution_start_time = time.perf_counter()
        if self.server.client_id is not None:
            self.server.send_sync("execution_start", {"prompt_id": prompt_id}, self.server.client_id)

        with torch.inference_mode():
            # delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d

            next_nodes = get_next_nodes_map(prompt)
            worklist_output_delete_if_changed(prompt, self.old_prompt, self.outputs, next_nodes)

            current_outputs = set(self.outputs.keys())
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            if self.server.client_id is not None:
                self.server.send_sync("execution_cached", {"nodes": list(current_outputs), "prompt_id": prompt_id},
                                      self.server.client_id)

            to_execute = worklist_will_execute(prompt, self.outputs, execute_outputs)

            # This call shouldn't raise anything if there's an error deep in
            # the actual SD code, instead it will report the node where the
            # error was raised
            executed, success, error, ex = worklist_execute(self.server, prompt, self.outputs, extra_data, prompt_id,
                                                            self.outputs_ui, to_execute, next_nodes)
            if success is not True:
                self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])
            self.server.last_node_id = None
            if self.server.client_id is not None:
                self.server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, self.server.client_id)

        print("Prompt executed in {:.2f} seconds".format(time.perf_counter() - execution_start_time))
        gc.collect()
        comfy.model_management.soft_empty_cache()
