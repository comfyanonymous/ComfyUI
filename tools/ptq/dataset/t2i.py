import torch
import os

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, calib_data_path="../data/calib_prompts.txt"):
        if not os.path.exists(calib_data_path):
            raise FileNotFoundError
        with open(calib_data_path, "r", encoding="utf8") as file:
            lst = [line.rstrip("\n") for line in file]
        self.prompts = lst

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
