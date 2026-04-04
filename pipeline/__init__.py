import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
import tempfile
import PIL.Image


class BaselinePipeline:
    def __init__(self, device='cuda', dtype=torch.float32):
       self.device = device
       self.dtype = dtype

    def __call__(self, batch, **kwargs):
        final_output = None
        return final_output

    def batch_preprocess(self, batch):
        """
        Processes the raw batch into a flattened BF (Batch*Frames) format
        ready for the Specific Pipeline.
        """
        processed_batch = batch
        return processed_batch