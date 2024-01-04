


class IfConotrolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "state": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ()
    FUNCTION = "execute"
    
    FLOW_INPUTS = [("FROM", "FLOW")]
    FLOW_OUTPUTS = [("TO", "FLOW"), ("On True", "FLOW"), ("On False", "FLOW")]
    FLOW_GOTO = "goto"      # goto function

    CATEGORY = "flow"

    def execute(self, state):
        return ()
    
    def goto(self, state, flows):
        """
        Go to which node.
        
        INPUTS:
        flow_outputs, tuple, the output of execute function
        flows, list, all possible flows: [goto which after the whole node, slot 0, slot 1, ...]
        
        RETURN:
        str or None, the id of the next node
        """
        # 
        if flows is None or len(flows) == 0:
            return None

        # 
        goto_idx = 1 if state else 2
        print(f"goto_idx: {goto_idx}")
        
        # decide go to which branch
        goto_id = None
        if len(flows) <= goto_idx:
            print("goto idx outof range.")
            goto_id = None
        else:
            print(f"goto id: {flows[goto_idx]}")
            goto_id = flows[goto_idx][0] if flows[goto_idx] is not None else None
        
        # no brach is valid, go to the default branch
        if goto_id is None:
            print(f"goto id is None, use flows[0]: {flows[0]}")
            goto_id = flows[0][0] if flows[0] is not None else None
        return goto_id
    
    
    
class MergeFlowNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ()
    FUNCTION = "execute"

    FLOW_INPUTS = [("FROM0", "FLOW"), ("FROM1", "FLOW")]
    FLOW_OUTPUTS = [("TO", "FLOW")]
    FLOW_GOTO = "goto"
    
    CATEGORY = "flow"
    
    def execute(self):
        return ()
        
    
    def goto(self, flows):
        """
        Go to which node.
        
        INPUTS:
        outputs, tuple, the output of execute function
        flows, list, all possible flows: [goto which after the whole node, slot 0, slot 1, ...]
        
        RETURN:
        str or None, the id of the next node
        """
        return flows[0][0] if (flows is not None and len(flows) > 0 and flows[0] is not None) else None
        
        
        
class AnyFirstValidInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "input0": ("ANY_DATA",)
                }, 
                "optional": {
                    "input1": ("ANY_DATA",),
                    "input2": ("ANY_DATA",),
                    "input3": ("ANY_DATA",),
                    "input4": ("ANY_DATA",),
                    "input5": ("ANY_DATA",)
                }
                }
    
    RETURN_TYPES = ("ANY_DATA", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    # def __init__(self) -> None:
    #     self.input_vals = [None, None, None, None, None, None]
    #     # self.input_valid = [False, False, False, False, False, False]
    
    def execute(self, input0, input1=None, input2=None, input3=None, input4=None, input5=None):
        input_list=[input0, input1, input2, input3, input4, input5]
        res = next((x for x in input_list if x is not None), None)
        return (res, )
    
    

class LoopFlowNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"nb_iteration": ("INT", {"default": 1})}}
    
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("current iter", )
    FUNCTION = "execute"
    
    FLOW_INPUTS = [("FROM", "FLOW")]
    FLOW_OUTPUTS = [("TO", "FLOW"), ("Loop Body", "FLOW")]
    FLOW_GOTO = "goto"      # goto function

    CATEGORY = "flow"
    
    
    def init(self):
        self.in_loop = True
        self.cur_item = -1

    
    def is_loop_end(self):
        return not self.in_loop
    
    
    def execute(self, nb_iteration):
        if self.in_loop:
            self.cur_item += 1
            if self.cur_item >= nb_iteration:
                self.in_loop = False
            print(f"[LoopNode] execute, cur_item: {self.cur_item}, nb_iter: {nb_iteration}")
            return (self.cur_item,)
        print(f"[LoopNode] not execute, cur_item: {self.cur_item}, nb_iter: {nb_iteration}")
        return (self.cur_item,)
    
    
    
    def goto(self, nb_iteration, flows):
        """
        Go to which node.
        
        INPUTS:
        flow_outputs, tuple, the output of execute function
        flows, list, all possible flows: [goto which after the whole node, slot 0, slot 1, ...]
        
        RETURN:
        str or None, the id of the next node
        """
        
        if self.in_loop:
            return flows[1][0] if (flows is not None and len(flows) > 1 and flows[1] is not None) else None
        else:
            return flows[0][0] if (flows is not None and len(flows) > 0 and flows[0] is not None) else None
    
    
    
    
NODE_CLASS_MAPPINGS = {
    
    "IfConotrolNode": IfConotrolNode,
    "MergeFlowNode": MergeFlowNode,
    "LoopFlowNode": LoopFlowNode,
    "AnyFirstValidInput": AnyFirstValidInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IfConotrolNode": "If(Go To)",
    "MergeFlowNode": "Merge Flow",
    "LoopFlowNode": "Loop",
    "AnyFirstValidInput": "Any First Valid Input"
}  