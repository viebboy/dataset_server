from torchvision.datasets import CIFAR10
from dataset_server import AsyncDataLoader
from tqdm import tqdm
import numpy as np
import time
from loguru import logger

class Dataset(CIFAR10):
    """
    Bootstrapping CIFAR10 dataset class from torchvision to be keyword argument based dataset
    """
    def __init__(self, **kwargs):
        super().__init__(root='./data/', train=kwargs['train'], download=True)

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        x = np.asarray(x)
        return x, y


if __name__ == '__main__':
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
    )

    start = time.time()
    for _ in range(10):
        for samples in tqdm(dataloader):
            pass
    stop = time.time()
    dataloader.close()
    logger.info(f'took {stop - start} seconds to loop through 10 epochs')
