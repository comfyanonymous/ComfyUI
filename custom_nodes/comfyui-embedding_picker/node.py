from pathlib import Path

import folder_paths


class EmbeddingPicker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        embeddings = folder_paths.get_filename_list("embeddings")

        return {
            "required": {
                "embedding": ((embeddings),),
                "emphasis": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                    },
                ),
                "append": (
                    "BOOLEAN",
                    {"default": False, "label_on": "true ", "label_off": "false "},
                ),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat_embedding"
    OUTPUT_NODE = False

    CATEGORY = "utils"

    def concat_embedding(self, text, embedding, emphasis, append):
        if emphasis < 0.05:
            return (text,)

        emb = "embedding:" + Path(embedding).stem

        emphasis = f"{emphasis:.3f}"
        if emphasis != "1.000":
            emb = f"({emb}:{emphasis})"

        output = f"{text}, {emb}" if append else f"{emb}, {text}"

        return (output,)
