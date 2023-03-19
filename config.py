import os
import torch


class Config:
    def __init__(self):
        self.data_path = "./squad"
        self.train_path = os.path.join(self.data_path, "train-v1.1.json")
        self.dev_path = os.path.join(self.data_path, "dev-v1.1.json")
        self.model_path = "./model"
        self.hidden_size = 768
        self.max_seq_length = 512
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
