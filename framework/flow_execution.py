import os
import sys
import copy
import json
import logging
import threading
import heapq
import traceback
import gc

import torch
import nodes

import comfy.model_management
from framework.app_log import LogUtils
from framework.flow_control_nodes import LoopFlowNode


def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)
    
    
def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__
    
    
class ExecuteContextStorage:
    def __init__(self) -> None:
        
        self.old_outputs = {}       # outputs during the last execution
                                    # {id: [list of outputs]}
        self.outputs = {}           # output during this execution
        self.is_changed = {}        # output is changed or not
        self.is_changed_hash = {}
        
        self.old_outputs_ui = {}    # 
        self.outputs_ui = {}

        # self.old_objects = {}
        self.objects = {}

        self.old_prompt = {}
        self.prompt = {}
        self.prompt_id = 0
        
        self.flows = {}
        self.extra_data = {}
        
        self.executed = set()
        
        
        
    def is_connection_input(self, node_prompt, input_name):
        """
        Check an input is a connection input or not.
        
        INPUT:
        node_prompt, prompt info of the node
        input_name, name of the input
        
        RETURN:
        True, the input is a connection input
        False, the input isn't a connection input or the input not found.
        """
        input_linked = node_prompt['is_input_linked']
        if input_name in input_linked:
            return input_linked[input_name]
        return False
        
        
        
    def get_object(self, node_id):
        class_type = self.prompt[node_id]['class_type']
        obj = self.objects.get((node_id, class_type), None)
        if obj is None:
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            obj = class_def()
            self.objects[(node_id, class_type)] = obj
            
        return obj
    
    
    def get_inputs(self, node_id):
        """
        Get Current inputs of a node
        
        INPUT:
        node_id, the id of the node
        RETURN:
        list, list of all inputs of the node
        """
        inputs = self.prompt[node_id]['inputs']
        class_type = self.prompt[node_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        valid_inputs = class_def.INPUT_TYPES()
        
        # print(f"[GetInputs] node id: {node_id}, {self.prompt[node_id]['class_type']}")
        # print(f"[GetInputs] inputs: {LogUtils.visible_convert(inputs)}")
        # print(f"[GetInputs] cur outputs: {LogUtils.visible_convert(self.outputs)}")
        
        input_datas = {}
        for input_name, input_val in inputs.items():
            
            if self.is_connection_input(node_prompt=self.prompt[node_id], input_name=input_name):
                original_output = self.outputs[input_val[0]][input_val[1]]
                input_datas[input_name] = original_output
                
            else:
                if ("required" in valid_inputs and input_name in valid_inputs["required"]) or ("optional" in valid_inputs and input_name in valid_inputs["optional"]):
                    input_datas[input_name] = input_val
                    
        if "hidden" in valid_inputs:
            h = valid_inputs["hidden"]
            for x in h:
                if h[x] == "PROMPT":
                    input_datas[x] = self.prompt
                if h[x] == "EXTRA_PNGINFO":
                    if "extra_pnginfo" in self.extra_data:
                        input_datas[x] = self.extra_data['extra_pnginfo']
                if h[x] == "UNIQUE_ID":
                    input_datas[x] = node_id
                
        return input_datas
    
    
    def get_old_output(self, node_id):
        node_out = self.old_outputs.get(node_id, [])
        node_out_ui = self.old_outputs_ui.get(node_id, [])
        return node_out, node_out_ui
    
        
    def save_outputs(self, node_id, node_outputs, node_output_uis, changed):
        self.outputs[node_id] = node_outputs
        self.outputs_ui[node_id] = node_output_uis
        self.is_changed[node_id] = changed
    
    
    def _prepare_objects(self):
        """
        Clear objects which are no longer needed
        """
        
        if len(self.objects) > 100 and len(self.objects) > len(self.prompt) * 2:
            to_delete = []
            for id_type in self.objects:
                node_id = id_type[0]
                node_type = id_type[1]
                
                useless = True
                if node_id in self.prompt:
                    if node_type == self.prompt[node_id].get('class_type', ''):
                        useless = False
                        
                if useless:
                    to_delete.append(id_type)
                    
            for id_type in to_delete:
                del self.objects[id_type]
                
        return self.objects
                
        
    def _prepare_old_outputs(self):
        to_delete = []
        for node_id in self.old_outputs:
            if node_id not in self.prompt:
                to_delete.append(node_id)
        for node_id in to_delete:
            node_out = self.old_outputs.pop(node_id)
            del node_out
            
            # remove from old_outputs_ui
            if node_id in self.old_outputs_ui:
                node_out_ui = self.old_outputs_ui.pop(node_id)
                del node_out_ui
                
            # remove from is_changed_hash
            if node_id in self.is_changed_hash:
                node_is_changed_hash = self.is_changed_hash.pop(node_id)
                del node_is_changed_hash
            
        return self.old_outputs
    
        
    def prepare_execution(self, prompt_id, new_prompt, flows, extra_data):
        self.prompt = new_prompt
        self.prompt_id = prompt_id
        self.flows = flows
        self.extra_data = extra_data
        
        self.outputs = {}
        self.is_changed = {}
        self.outputs_ui = {}
        self.executed = set()
        
        # prepare outputs
        self._prepare_old_outputs()
        
        # prepare objects
        self._prepare_objects()
        
    
    def cleanup_execution(self):
        self.old_outputs = self.outputs
        self.outputs = {}
        
        self.old_outputs_ui = self.outputs_ui
        self.outputs_ui = {}
        
        # self.old_objects = self.objects
        # self.objects = {}
        
        # self.old_prompt = self.prompt
        # self.prompt = {}
        
        self.old_prompt = {}
        for node_id in self.executed:
            if node_id in self.prompt:
                self.old_prompt[node_id] = self.prompt[node_id]
            
        
        
        
        
class SequenceFlow:
    def __init__(self, first_node_id, context:ExecuteContextStorage, server) -> None:
        self.context = context
        self.first_node_id = first_node_id
        self.server = server
        
    
    def _on_node_executing(self, node_id):
        print(f"node executing")
        if self.server.client_id is not None:
            self.server.last_node_id = node_id
            self.server.send_sync("executing", { "node": node_id, "prompt_id": self.context.prompt_id }, self.server.client_id)

    def _on_node_executed(self, node_id, node_output_ui=[]):
        # 
        if len(node_output_ui) > 0:
            if self.server.client_id is not None:
                self.server.send_sync("executed", { "node": node_id, "output": node_output_ui, "prompt_id": self.context.prompt_id }, self.server.client_id)
        
        
    def _is_inputs_changed(self, node_id, obj, input_datas):
        """
        Wether the inputs of a node changed or not.
        
        INPUT:
        node_id, the id of the node
        
        RETURN:
        True, the input changed
        False, the input not changed
        """
        node_inputs = self.context.prompt[node_id]['inputs']
        old_inputs = self.context.old_prompt[node_id]['inputs'] if node_id in self.context.old_prompt else {}
        is_changed = False
        
        # IS_CHANGED defined
        class_type = self.context.prompt[node_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        if hasattr(class_def, 'IS_CHANGED'):
            is_changed_hash = getattr(obj, "IS_CHANGED")(**input_datas)
            print(f"[IsInputChanged] is changed hash: {is_changed_hash}")
            
            old_hash = self.context.is_changed_hash.get(node_id, '')
            if is_changed_hash != old_hash:
                is_changed = True
                self.context.is_changed_hash[node_id] = is_changed_hash
                print(f"[IsInputChanged] input changed: is_changed_hash NOT the same.")
        
        # check all inputs
        if is_changed == False:
            for input_name in node_inputs:
                is_connection = self.context.is_connection_input(node_prompt=self.context.prompt[node_id], input_name=input_name)
                
                if is_connection:
                    #find connected original node
                    original_node_id = node_inputs[input_name][0]
                    is_cur_input_changed = self.context.is_changed[original_node_id]
                    if is_cur_input_changed:
                        print(f"[IsInputChanged] input changed: input NODE changed.")
                else:
                    if input_name in old_inputs:
                        is_cur_input_changed = (node_inputs[input_name] != old_inputs[input_name])
                        if is_cur_input_changed:
                            print(f"[IsInputChanged] input changed: input value changed.")
                    else:
                        is_cur_input_changed = True
                        if is_cur_input_changed:
                            print(f"[IsInputChanged] input changed: previous input not exist.")
                    
                # input changed 
                if is_cur_input_changed:
                    is_changed = True
                    break

        # previous node outputs do not exist
        if is_changed == False and node_id not in self.context.old_outputs:
            is_changed = True
            if is_changed:
                print(f"[IsInputChanged] input changed: previous output not exist.")
                  
        return is_changed
    
    
    def _execute_node(self, obj, input_datas, allow_interrupt=True):
        """
        Parse output datas from executed results
        
        INPUTS:
        obj, the node object
        
        RETURN:
        list, list of the results.
        """
        results = []
        uis = []
        
        input_is_list = False
        if hasattr(obj, "INPUT_IS_LIST"):
            input_is_list = obj.INPUT_IS_LIST
        
        if allow_interrupt:
            nodes.before_node_execution()
        if input_is_list:
            for key,val in input_datas.items():
                input_datas[key] = [val]
        return_values = getattr(obj, obj.FUNCTION)(**input_datas)
        # print(f"[Execute Node] executed result: {LogUtils.visible_convert(return_values)}")
        
        # return_values = [return_values]
        # for r in return_values:
        #     if isinstance(r, dict):
        #         if 'ui' in r:
        #             uis.append(r['ui'])
        #         if 'result' in r:
        #             results.append(r['result'])
        #     else:
        #         results.append(r)
        

        
        # output = []
        # if len(results) > 0:
        #     # check which outputs need concatenating
        #     output_is_list = [False] * len(results[0])
        #     if hasattr(obj, "OUTPUT_IS_LIST"):
        #         output_is_list = obj.OUTPUT_IS_LIST

        #     # merge node execution results
        #     for i, is_list in zip(range(len(results[0])), output_is_list):
        #         if is_list:
        #             output.append([x for o in results for x in o[i]])
        #         else:
        #             output.append([o[i] for o in results])
        # output = output[0] if len(output) > 0 else None

        # ui = dict()    
        # if len(uis) > 0:
        #     ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
        
        
        if isinstance(return_values, dict):
            if 'ui' in return_values:
                uis = return_values['ui']
            if 'result' in return_values:
                results = return_values['result']
        else:
            results = return_values
        output = results
        ui = uis
            
        # print(f"[Execute Node] final output: {LogUtils.visible_convert(output)}")
        # print(f"[Execute Node] final ui: {ui}")
        
        return output, ui
        
        
    def _handle_flow_control_node(self, node_id, obj, input_datas):
        """
        Handle flow control node before executing.
        
        RETURN:
        (bool, str)
        bool, is this node a flow-control node
        str, next_node_id, the id of the next node.
        """
        node_type = self.context.prompt[node_id]['class_type']
        # if isinstance(obj, IfConotrolNode):
        if node_type == "IfConotrolNode" or node_type == "MergeFlowNode":
            # normalout, branch_bool = getattr(obj, obj.FUNCTION)(**input_datas)
            outputs, uis = self._execute_node(obj, input_datas, True)
            node_flows = self.context.flows[node_id]
            if hasattr(obj, 'FLOW_GOTO'):
                next_node_id = getattr(obj, obj.FLOW_GOTO)(**input_datas, flows = node_flows)
            else:
                next_node_id = self._node_default_goto(node_flows)
            print(f"[Handle control node] is control node.")
            return (True, next_node_id, True, None, None)
        elif node_type == "LoopFlowNode":
            # Loop Flow
            loop_flow = LoopFlow(node_id, self.context, self.server)
            # init loop flow
            loop_flow.init_loop(node_id, obj)
            # execute loop
            succ, err_detail, exp = loop_flow.execute()
            
            # next_node_id 
            next_node_id = loop_flow.goto()
            return (True, next_node_id, succ, err_detail, exp)
             
        else:
            print(f"[Handle control node] NOT control node.")
            return (False, None, True, None, None)
    
    
    def _node_default_goto(self, flows):
        return flows[0][0] if (flows is not None and len(flows) > 0 and flows[0] is not None) else None
    
    
    def _get_next_node(self, node_id):
        goto_infos = self.context.flows.get(node_id, None)
        if goto_infos is not None and len(goto_infos) > 0:
            goto_id = goto_infos[0][0] if goto_infos[0] is not None else None
        else:
            goto_id = None
        return goto_id
    
    
    def _collect_error_details(self, node_id, input_datas, exp):
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_datas is not None:
            input_data_formatted = {}
            for name, inputs in input_datas.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in self.context.outputs.items():
            output_data_formatted[node_id] = [format(l) for l in node_outputs]

        logging.error("!!! Exception during processing !!!")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": node_id,
            "exception_message": str(exp),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted
        }
    
        
    def execute(self):
        """
        Execute the flow
        
        RETURN:
        bool, successful or not
        dict, error details
        exception, exceptions
        """
        cur_node_id = self.first_node_id
        
        # while this is valid id
        while cur_node_id is not None:
            
            print(f"[Execute Node] ****************** {cur_node_id}, {self.context.prompt[cur_node_id]['class_type']}")
            
            
            input_datas = None
            try:
                self._on_node_executing(cur_node_id)
                
                # get node object
                obj = self.context.get_object(cur_node_id)
                
                # get inputs
                input_datas = self.context.get_inputs(cur_node_id)
                print(f"[Execute Node] inputs: {LogUtils.visible_convert(input_datas)}")
                
                is_control_node, next_node_id, subflow_succ, err_detail, exp = self._handle_flow_control_node(cur_node_id, obj, input_datas)
                
                node_output_ui = []
                # not a control node, just execute it.
                if not is_control_node:
                    # input changed, execute this node
                    if self._is_inputs_changed(cur_node_id, obj, input_datas):
                        print(f"[Execute Node] input changed.")
                        
                        # execute
                        node_output, node_output_ui = self._execute_node(
                                    obj=obj, input_datas=input_datas, allow_interrupt=True)
                        
                        # save output
                        self.context.save_outputs(cur_node_id, node_output, node_output_ui, True)

                    # input not change, copy the result and skip this node
                    else:
                        print(f"[Execute Node] input NOT changed.")
                        node_output, node_output_ui = self.context.get_old_output(cur_node_id)
                        self.context.save_outputs(cur_node_id, node_output, node_output_ui, False)
                    
                    print(f"[Execute Node] output: {LogUtils.visible_convert(node_output)}")
                    print(f"[Execute Node] output ui: {LogUtils.visible_convert(node_output_ui)}")
                    
                    # go to the next node
                    next_node_id = self._get_next_node(cur_node_id)
                #     print(f"[Execute Node] next id: {next_node_id}")
                
                # is a control node.
                else:
                    if not subflow_succ:
                        print(f"[Execute Node] Sub flow executed FAILED: {err_detail}")
                        return (subflow_succ, err_detail, exp)
                    
                
                # add to executed list
                self.context.executed.add(cur_node_id)
                # 
                self._on_node_executed(cur_node_id, node_output_ui)
                
                cur_node_id = next_node_id
                print(f"[Execute Node] next id: {next_node_id}")

            
            except comfy.model_management.InterruptProcessingException as iex:
                logging.info("Processing interrupted")
                # skip formatting inputs/outputs
                error_details = {
                    "node_id": cur_node_id,
                }
                print(f"[Execute Node] error: {iex}")
                return (False, error_details, iex)
            
            except Exception as ex:
                error_details = self._collect_error_details(cur_node_id, input_datas, ex)
                return (False, error_details, ex)
                
        
        print(f"[Execute Node] Done")
        return (True,None, None) 
      
        

class LoopFlow(SequenceFlow):
    def __init__(self, first_node_id, context: ExecuteContextStorage, server) -> None:
        super().__init__(first_node_id, context, server)
        
        self.loop_node = None
        self.loop_node_id = None
        
        
    
    def init_loop(self, loop_node_id, loop_node_obj:LoopFlowNode):
        self.loop_node = loop_node_obj
        self.loop_node_id = loop_node_id
        self.loop_node.init()

 
    def execute(self):
        """
        Execute the flow
        
        RETURN:
        bool, successful or not
        dict, error details
        exception, exceptions
        """
        print(f"[LoopFlow] execute start.")
        while not self.loop_node.is_loop_end():
            input_datas = None
            try:
                # start executing loop node
                self._on_node_executing(self.loop_node_id)
                # get inputs
                input_datas = self.context.get_inputs(self.loop_node_id)
                node_output, node_output_ui = self._execute_node(
                                        obj=self.loop_node, input_datas=input_datas, allow_interrupt=True)
                # save output
                self.context.save_outputs(self.loop_node_id, node_output, node_output_ui, True)
                            
                self.context.executed.add(self.loop_node_id)
                # 
                self._on_node_executed(self.loop_node_id, [])
                
                print(f"[Execute Node] output: {LogUtils.visible_convert(node_output)}")
                print(f"[Execute Node] output ui: {LogUtils.visible_convert(node_output_ui)}")
                
                if self.loop_node.is_loop_end():
                    print(f"[LoopFlow] loop end.")
                    return (True, None, None)
                
                self.first_node_id = self.loop_node.goto(**input_datas, flows=self.context.flows.get(self.loop_node_id, None))
                
                print(f"[LoopFlow] first_node of loop: {self.first_node_id}")
                
            except comfy.model_management.InterruptProcessingException as iex:
                logging.info("Processing interrupted")
                # skip formatting inputs/outputs
                error_details = {
                    "node_id": self.loop_node_id,
                }
                print(f"[Execute Node] error: {iex}")
                return (False, error_details, iex)
            
            except Exception as ex:
                error_details = self._collect_error_details(self.loop_node_id, input_datas, ex)
                return (False, error_details, ex)
                
            # execute loop body
            succ, err_details, exp = super().execute()
            
            if not succ:
                print(f"[LoopNode] Error during executing loop body.")
                return (succ, err_details, exp)
        
        print(f"[LoopFlow] loop end.")
        return (True, None, None)
            
            
    def goto(self):
        input_datas = self.context.get_inputs(self.loop_node_id)
        return self.loop_node.goto(**input_datas, flows=self.context.flows.get(self.loop_node_id, None))
 


class FlowExecutor:
    def __init__(self, server):
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.old_prompt = {}
        self.server = server
        self.context = ExecuteContextStorage()
        

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

    def execute(self, prompt, prompt_id, flows, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)
        
        print(f"[Execute] prompt: {prompt}")        
        print(f"[Execute] prompt id: {prompt_id}")
        print(f"[Execute] flows: {flows}")
        print(f"[Execute] extra data: {extra_data}")
        print(f"[Execute] execute outputs: {execute_outputs}")

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        if self.server.client_id is not None:
            self.server.send_sync("execution_start", { "prompt_id": prompt_id}, self.server.client_id)

        with torch.inference_mode():
            
            # prepare execution context
            self.context.prepare_execution(prompt_id=prompt_id, new_prompt=prompt, flows=flows, extra_data=extra_data)
            
            # cleanup models
            comfy.model_management.cleanup_models()
            
            # in degree
            in_degree = {}
            for from_id, goto_list in flows.items():
                if goto_list is None:
                    continue
                for to_info in goto_list:
                    if to_info is not None:
                        to_id = to_info[0]
                        val = in_degree.get(to_id, 0)
                        val+=1
                        in_degree[to_id] = val
            
            # find the first nodes
            first_node_ids = []
            for node_id in prompt:
                if node_id not in in_degree or in_degree[node_id] <=0:
                    first_node_ids.append(node_id)
                 
            print(f"[Execute] first nodes: {first_node_ids}") 
            # exexute each flow  
            for first_node_id in first_node_ids:
                # 
                flow_exe = SequenceFlow(first_node_id=first_node_id, context=self.context, server= self.server)
                
                # execute
                succ, err, ex = flow_exe.execute()
                
                if succ is not True:
                    self.handle_execution_error(prompt_id, prompt, self.context.outputs, self.context.executed, err, ex)
                    break
             
            print(f"[Execution] output list: {LogUtils.visible_convert(self.context.outputs)}")   
            print(f"[Execution] execution DONE.")

            self.server.last_node_id = None
            
            # context cleanup
            self.context.cleanup_execution()
