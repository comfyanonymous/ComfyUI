import random


class TestGenerator:

    def __init__(self):
        self.testID = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "hidden": {
                "testId": ("STRING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "test_generator"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    TESTID = 0
    @classmethod
    def IS_CHANGED(s, clip, testId=None):
        # intValue = random.randint(0, 100)
        # value = str(intValue)
        if TestGenerator.TESTID < 2:
            TestGenerator.TESTID += 1
        return str(TestGenerator.TESTID)

    def test_generator(self, clip, testId=None):
        tokens = clip.tokenize("test")
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

NODE_CLASS_MAPPINGS = {
    "TestGenerator": TestGenerator
}