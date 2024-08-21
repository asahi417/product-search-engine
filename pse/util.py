import logging
from gc import collect
import torch
import numpy as np


def clear_cache():
    torch.cuda.empty_cache()
    collect()


def np_save(array: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        np.save(f, array)


def np_load(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger
