from datasets import load_dataset
from torch.utils.data import Dataset


class OmniDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train").filter(
            lambda x: x["instruction_language"] == "en")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "prompt": sample["instruction"],
            "img_pil": sample["input_image_raw"]
        }


if __name__ == "__main__":
    dataset = OmniDataset()
    dataset.__getitem__(0)
