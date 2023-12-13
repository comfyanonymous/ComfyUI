


class IfConotrolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "state": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("FLOW", "FLOW")
    RETURN_NAMES = ("On True", "On False")
    FUNCTION = "execute"

    CATEGORY = "flow/if"

    def execute(self, state):
        if state:
            return (True, False)
        else:
            
            return (False, True)
    
    def goto(self, outputs, flows):
        """
        Go to which node.
        
        INPUTS:
        outputs, tuple, the output of execute function
        flows, list, all possible flows: [goto which after the whole node, slot 0, slot 1, ...]
        
        RETURN:
        str or None, the id of the next node
        """
        # 
        if flows is None or len(flows) == 0:
            return None

        goto_idx = None
        for slot in range(len(outputs)):
            if outputs[slot]:
                goto_idx = slot + 1 # the first one is the flow of the whole node
                break
            
        print(f"goto_idx: {goto_idx}")
        goto_id = None
        if len(flows) <= goto_idx:
            print("goto idx outof range.")
            goto_id = None
        else:
            print(f"goto id: {flows[goto_idx]}")
            goto_id = flows[goto_idx]
        
        if goto_id is None:
            print(f"goto id is None, use flows[0]: {flows[0]}")
            goto_id = flows[0]
        return goto_id
    
    
NODE_CLASS_MAPPINGS = {
    
    "IfConotrolNode": IfConotrolNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IfConotrolNode": "If(Go To)",
}  