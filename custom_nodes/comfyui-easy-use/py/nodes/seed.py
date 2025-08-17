from ..config import MAX_SEED_NUM
import hashlib
import random

class easySeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Seed"

    def doit(self, seed=0, prompt=None, extra_pnginfo=None, my_unique_id=None):
        return seed,

class seedList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_num": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                "max_num": ("INT", {"default": MAX_SEED_NUM, "min": 0 }),
                "method": (["random", "increment", "decrement"], {"default": "random"}),
                "total": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM,}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "total")
    FUNCTION = "doit"
    DESCRIPTION = "Random number seed that can be used in a for loop, by connecting index and easy indexAny node to realize different seed values in the loop."

    CATEGORY = "EasyUse/Seed"

    def doit(self, min_num, max_num, method, total, seed=0, prompt=None, extra_pnginfo=None, my_unique_id=None):
        random.seed(seed)

        seed_list = []
        if min_num > max_num:
            min_num, max_num = max_num, min_num
        for i in range(total):
            if method == 'random':
                s = random.randint(min_num, max_num)
            elif method == 'increment':
                s = min_num + i
                if s > max_num:
                    s = max_num
            elif method == 'decrement':
                s = max_num - i
                if s < min_num:
                    s = min_num
            seed_list.append(s)
        return seed_list, total

    @classmethod
    def IS_CHANGED(s, seed, **kwargs):
        m = hashlib.sha256()
        m.update(seed)
        return m.digest().hex()

# 全局随机种
class globalSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "control_before_generate", "label_off": "control_after_generate"}),
                "action": (["fixed", "increment", "decrement", "randomize",
                            "increment for each node", "decrement for each node", "randomize for each node"], ),
                "last_seed": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Seed"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}


NODE_CLASS_MAPPINGS = {
    "easy seed": easySeed,
    "easy seedList": seedList,
    "easy globalSeed": globalSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy seed": "EasySeed",
    "easy seedList": "EasySeedList",
    "easy globalSeed": "EasyGlobalSeed",
}