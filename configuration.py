import argparse
from pathlib import Path
import os

class Config:
    """
    Config: Get model parameters for modifications
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--train_csv", type=str,
                                    default=os.path.join("data", "Train-word.csv"),
                                     help='Path to train directory')
        self.parser.add_argument("--test_csv", type=str,
                                    default=os.path.join("data", "Test-word.csv"),
                                     help='Path to test directory')
        self.parser.add_argument("--dev_csv", type=str,
                                    default=os.path.join("data", "Val-word.csv"),
                                     help='Path to validation directory')
        self.parser.add_argument("--cuda", type=bool, default=True, help='Cuda Utilization')
        self.parser.add_argument("--learning_rate", type=float, default=1e-5, help='Learning Rate')
        self.parser.add_argument("--max_grad_norm", type=float, default=1e-05, help='Maximum Grad Norm')
        self.parser.add_argument("--pretrained_model", type=str)
        self.parser.add_argument("--prediction_path", type=str,
                                    default=os.path.join("/assets/predictions"),
                                     help='Path to save results')
        self.parser.add_argument("-f")
                                 
    def get_re_args(self):
        """
          Return parser
        """
        self.parser.add_argument("--pretrained_new", type=str,
                                    default=os.path.join("/assets/checkpoints"),
                                     help='Path to trained model')
        self.parser.add_argument("--max_len", type=int, default=128, help='Max Lenght')
        self.parser.add_argument("--epochs", type=int, default=10, help='Training Epochs')
        self.parser.add_argument("--train_batch_size", type=int, default=32, help='Train Batch Size')
        self.parser.add_argument("--valid_batch_size", type=int, default=16, help='Dev/Test Batch Size')
        return self.parser.parse_args()