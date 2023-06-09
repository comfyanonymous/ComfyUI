import os
import sys
import copy
import json
import threading
import heapq
import traceback
import gc
import time
import itertools

import torch
import nodes

import comfy.model_management


def get_input_data_batches(input_data_all):
    """Given input data that can contain combinatorial input values, returns all
    the possible batches that can be made by combining the different input
    values together."""

    input_to_index = {}
    index_to_values = []

    # Sort by input name first so the order which batch inputs are applied can
    # be easily calculated (node execution order first, then alphabetical input
    # name second)
    sorted_input_names = sorted(input_data_all.keys())

    i = 0
    for input_name in sorted_input_names:
        value = input_data_all[input_name]
        if isinstance(value, dict) and "combinatorial" in value:
            input_to_index[input_name] = i
            index_to_values.append(value["values"])
            i += 1

    if len(index_to_values) == 0:
        # No combinatorial options.
        return [input_data_all]

    batches = []

    for combination in list(itertools.product(*index_to_values)):
        batch = {}
        for input_name, value in input_data_all.items():
            if isinstance(value, dict) and "combinatorial" in value:
                combination_index = input_to_index[input_name]
                batch[input_name] = [combination[combination_index]]
            else:
                # already made into a list by get_input_data
                batch[input_name] = value
        batches.append(batch)

    return batches

def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, extra_data={}):
    """Given input data from the prompt, returns a list of input data dicts for
    each combinatorial batch."""
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    for x in inputs:
        input_data = inputs[x]
        required_or_optional = ("required" in valid_inputs and x in valid_inputs["required"]) or ("optional" in valid_inputs and x in valid_inputs["optional"])
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                return None

            # This is a list of outputs for each batch of combinatorial inputs.
            # Without any combinatorial inputs, it's a list of length 1.
            outputs_for_all_batches = outputs[input_unique_id]

            def flatten(list_of_lists):
                return list(itertools.chain.from_iterable(list_of_lists))

            if len(outputs_for_all_batches) == 1:
                # Single batch, no combinatorial stuff
                input_data_all[x] = outputs_for_all_batches[0][output_index]
            else:
                # Make the outputs into a list for map-over-list use
                # (they are themselves lists so flatten them afterwards)
                input_values = [batch_output[output_index] for batch_output in outputs_for_all_batches]
                input_values = flatten(input_values)
                input_data_all[x] = input_values
        elif is_combinatorial_input(input_data):
            if required_or_optional:
                input_data_all[x] = { "combinatorial": True, "values": input_data["values"] }
        else:
            if required_or_optional:
                input_data_all[x] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [prompt]
            if h[x] == "EXTRA_PNGINFO":
                if "extra_pnginfo" in extra_data:
                    input_data_all[x] = [extra_data['extra_pnginfo']]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]

    input_data_all_batches = get_input_data_batches(input_data_all)

    return input_data_all_batches

def slice_lists_into_dict(d, i):
    """
    get a slice of inputs, repeat last input when list isn't long enough
    d={ "seed": [ 1, 2, 3 ], "steps": [ 4, 8 ] }, i=2 -> { "seed": 3, "steps": 8 }
    """
    d_new = {}
    for k, v in d.items():
        d_new[k] = v[i if len(v) > i else -1]
    return d_new

def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    # check if node wants the lists
    intput_is_list = False
    if hasattr(obj, "INPUT_IS_LIST"):
        intput_is_list = obj.INPUT_IS_LIST

    max_len_input = max([len(x) for x in input_data_all.values()])
     
    def format_dict(d):
        s = []
        for k,v in d.items():
            st = f"{k}: "
            if isinstance(v, list):
                st += f"list[len: {len(v)}]["
                i = []
                for v2 in v:
                    i.append(v2.__class__.__name__)
                st += ",".join(i) + "]"
            else:
                st += str(type(v))
            s.append(st)
        return "( " + ", ".join(s) + " )"

    max_len_input = max(len(x) for x in input_data_all.values())

    results = []
    if intput_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)(**input_data_all))
    else: 
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)(**slice_lists_into_dict(input_data_all, i)))
    return results

def get_output_data(obj, input_data_all_batches, server, unique_id, prompt_id):
    all_outputs = []
    all_outputs_ui = []
    total_batches = len(input_data_all_batches)

    for batch_num, batch in enumerate(input_data_all_batches):
        return_values = map_node_over_list(obj, batch, obj.FUNCTION, allow_interrupt=True)

        uis = []
        results = []

        for r in return_values:
            if isinstance(r, dict):
                if 'ui' in r:
                    uis.append(r['ui'])
                if 'result' in r:
                    results.append(r['result'])
            else:
                results.append(r)

        output = []
        if len(results) > 0:
            # check which outputs need concatenating
            output_is_list = [False] * len(results[0])
            if hasattr(obj, "OUTPUT_IS_LIST"):
                output_is_list = obj.OUTPUT_IS_LIST

            # merge node execution results
            for i, is_list in zip(range(len(results[0])), output_is_list):
                if is_list:
                    output.append([x for o in results for x in o[i]])
                else:
                    output.append([o[i] for o in results])

        output_ui = None
        if len(uis) > 0:
            output_ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}

        all_outputs.append(output)
        all_outputs_ui.append(output_ui)

        outputs_ui_to_send = None
        if any(all_outputs_ui):
            outputs_ui_to_send = all_outputs_ui

        # update the UI after each batch finishes
        if server.client_id is not None:
            message = {
                "node": unique_id,
                "output": outputs_ui_to_send,
                "prompt_id": prompt_id,
                "batch_num": batch_num,
                "total_batches": total_batches
            }
            server.send_sync("executed", message, server.client_id)

    return all_outputs, all_outputs_ui

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def recursive_execute(server, prompt, outputs, current_item, extra_data, executed, prompt_id, outputs_ui):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        return (True, None, None)

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                result = recursive_execute(server, prompt, outputs, input_unique_id, extra_data, executed, prompt_id, outputs_ui)
                if result[0] is not True:
                    # Another node failed further upstream
                    return result

    input_data_all = None
    try:
        input_data_all_batches = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)
        if server.client_id is not None:
            server.last_node_id = unique_id
            server.send_sync("executing", { "node": unique_id, "prompt_id": prompt_id, "total_batches": len(input_data_all_batches) }, server.client_id)
        obj = class_def()

        output_data_from_batches, output_ui_from_batches = get_output_data(obj, input_data_all_batches, server, unique_id, prompt_id)
        outputs[unique_id] = output_data_from_batches
        if any(output_ui_from_batches):
            outputs_ui[unique_id] = output_ui_from_batches
        elif unique_id in outputs_ui:
            outputs_ui.pop(unique_id)
    except comfy.model_management.InterruptProcessingException as iex:
        print("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": unique_id,
        }

        return (False, error_details, iex)
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
        return (False, error_details, ex)

    executed.add(unique_id)

    return (True, None, None)

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
    to_delete = False
    if hasattr(class_def, 'IS_CHANGED'):
        if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
            is_changed_old = old_prompt[unique_id]['is_changed']
        if 'is_changed' not in prompt[unique_id]:
            input_data_all_batches = get_input_data(inputs, class_def, unique_id, outputs)
            if input_data_all_batches is not None:
                 try:
                    #is_changed = class_def.IS_CHANGED(**input_data_all)
                    for batch in input_data_all_batches:
                        if map_node_over_list(class_def, batch, "IS_CHANGED"):
                            is_changed = True
                            break
                    prompt[unique_id]['is_changed'] = is_changed
                 except:
                    to_delete = True
        else:
            is_changed = prompt[unique_id]['is_changed']

    if unique_id not in outputs:
        return True

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
        self.outputs_ui = {}
        self.old_prompt = {}
        self.server = server

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

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
            self.server.send_sync("execution_start", { "prompt_id": prompt_id}, self.server.client_id)

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
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            if self.server.client_id is not None:
                self.server.send_sync("execution_cached", { "nodes": list(current_outputs) , "prompt_id": prompt_id}, self.server.client_id)
            executed = set()
            output_node_id = None
            to_execute = []

            for node_id in list(execute_outputs):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                #always execute the output that depends on the least amount of unexecuted nodes first
                to_execute = sorted(list(map(lambda a: (len(recursive_will_execute(prompt, self.outputs, a[-1])), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]

                # This call shouldn't raise anything if there's an error deep in
                # the actual SD code, instead it will report the node where the
                # error was raised
                success, error, ex = recursive_execute(self.server, prompt, self.outputs, output_node_id, extra_data, executed, prompt_id, self.outputs_ui)
                if success is not True:
                    self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)
                    break

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])
            self.server.last_node_id = None
            if self.server.client_id is not None:
                self.server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, self.server.client_id)

        print("Prompt executed in {:.2f} seconds".format(time.perf_counter() - execution_start_time))
        gc.collect()
        comfy.model_management.soft_empty_cache()


def is_combinatorial_input(val):
    return isinstance(val, dict) and "__inputType__" in val


def get_raw_inputs(raw_val):
    if isinstance(raw_val, list):
        # link to another node
        return [raw_val]
    elif is_combinatorial_input(raw_val):
        return raw_val["values"]
    return [raw_val]


def clamp_input(val, info, class_type, obj_class, x):
    errors = []

    if is_combinatorial_input(val):
        if len(val["values"]) == 0:
            error = {
                "type": "combinatorial_input_missing_values",
                "message": f"Combinatorial input has no values in its list.",
                "details": f"{x}",
                "extra_info": {
                    "input_name": x,
                    "input_config": info,
                    "received_value": val,
                }
            }
            return (False, None, error)
        for i, val_choice in enumerate(val["values"]):
            r = clamp_input(val_choice, info, class_type, obj_class, x)
            if r[0] == False:
                return r
            val["values"][i] = r[1]
        return (True, val, None)

    type_input = info[0]

    try:
        if type_input == "INT":
            val = int(val)
        if type_input == "FLOAT":
            val = float(val)
        if type_input == "STRING":
            val = str(val)
    except Exception as ex:
        error = {
            "type": "invalid_input_type",
            "message": f"Failed to convert an input value to a {type_input} value",
            "details": f"{x}, {val}, {ex}",
            "extra_info": {
                "input_name": x,
                "input_config": info,
                "received_value": val,
                "exception_message": str(ex)
            }
        }
        return (False, None, error)

    if len(info) > 1:
        if "min" in info[1] and val < info[1]["min"]:
            error = {
                "type": "value_smaller_than_min",
                "message": "Value {} smaller than min of {}".format(val, info[1]["min"]),
                "details": f"{x}",
                "extra_info": {
                    "input_name": x,
                    "input_config": info,
                    "received_value": val,
                }
            }
            return (False, None, error)
        if "max" in info[1] and val > info[1]["max"]:
            error = {
                "type": "value_bigger_than_max",
                "message": "Value {} bigger than max of {}".format(val, info[1]["max"]),
                "details": f"{x}",
                "extra_info": {
                    "input_name": x,
                    "input_config": info,
                    "received_value": val,
                }
            }
            return (False, None, error)

    return (True, val, None)


def validate_inputs(prompt, item, validated):
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    required_inputs = class_inputs['required']

    errors = []
    valid = True

    for x in required_inputs:
        if x not in inputs:
            error = {
                "type": "required_input_missing",
                "message": "Required input is missing",
                "details": f"{x}",
                "extra_info": {
                    "input_name": x
                }
            }
            errors.append(error)
            continue

        val = inputs[x]
        info = required_inputs[x]
        type_input = info[0]
        if isinstance(val, list):
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            if r[val[1]] != type_input:
                received_type = r[val[1]]
                details = f"{x}, {received_type} != {type_input}"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)
                continue
            try:
                r = validate_inputs(prompt, o_id, validated)
                if r[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = (False, reasons, o_id)
                continue
        else:
            r = clamp_input(val, info, class_type, obj_class, x)
            if r[0] == False:
                errors.append(r[2])
                continue
            else:
                inputs[x] = r[1]

            if hasattr(obj_class, "VALIDATE_INPUTS"):
                input_data_all_batches = get_input_data(inputs, obj_class, unique_id)
                #ret = obj_class.VALIDATE_INPUTS(**input_data_all)
                ret = map_node_over_list(obj_class, input_data_all, "VALIDATE_INPUTS")
                for batch in input_data_all_batches:
                    ret = map_node_over_list(obj_class, batch, "VALIDATE_INPUTS")
                    for r in ret:
                        if r != True:
                            details = f"{x}"
                            if r is not False:
                                details += f" - {str(r)}"

                            error = {
                                "type": "custom_validation_failed",
                                "message": "Custom validation failed for node",
                                "details": details,
                                "extra_info": {
                                    "input_name": x,
                                    "input_config": info,
                                    "received_value": val,
                                }
                            }
                            errors.append(error)
                            continue
            else:
                if isinstance(type_input, list):
                    # Account for more than one combinatorial value
                    raw_vals = get_raw_inputs(val)
                    for raw_val in raw_vals:
                        if raw_val not in type_input:
                            input_config = info
                            list_info = ""

                            # Don't send back gigantic lists like if they're lots of
                            # scanned model filepaths
                            if len(type_input) > 20:
                                list_info = f"(list of length {len(type_input)})"
                                input_config = None
                            else:
                                list_info = str(type_input)

                            error = {
                                "type": "value_not_in_list",
                                "message": "Value not in list",
                                "details": f"{x}: '{raw_val}' not in {list_info}",
                                "extra_info": {
                                    "input_name": x,
                                    "input_config": input_config,
                                    "received_value": raw_val,
                                }
                            }
                            errors.append(error)
                            continue

    if len(errors) > 0 or valid is not True:
        ret = (False, errors, unique_id)
    else:
        ret = (True, [], unique_id)

    validated[unique_id] = ret
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt):
    outputs = set()
    for x in prompt:
        class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE == True:
            outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], [])

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            print(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                print("* (prompt):")
                for reason in reasons:
                    print(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        print(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            print(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            print("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)


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
                self.history[prompt[1]]["outputs"][o] = outputs[o]
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
