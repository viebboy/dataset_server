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

def test_default():
    logger.info('test default setting')
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        qt_threading=False,
    )

    start = time.time()
    for samples in tqdm(dataloader):
        pass
    stop = time.time()
    dataloader.close()
    logger.info('complete testing the default config')

def test_cache_client():
    logger.info('test caching setting in client side')
    cache_setting = {
        'cache_side': 'client',
        'cache_prefix': './data/train_cache',
        'rewrite': True,
        'update_frequency': 3,
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        qt_threading=False,
        nearby_shuffle=100,
        cache_setting=cache_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing caching on client side')

def test_cache_server():
    logger.info('test caching setting in server side')
    cache_setting = {
        'cache_side': 'server',
        'cache_prefix': './data/train_cache',
        'rewrite': False,
        'update_frequency': 3,
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        qt_threading=False,
        nearby_shuffle=100,
        cache_setting=cache_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing caching on server side')

def test_rotation():
    logger.info('test rotation feature')
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        qt_threading=False,
        nearby_shuffle=100,
        rotation=3,
        min_rotation_size=10,
        max_rotation_size=1000,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation')

if __name__ == '__main__':
    #---------------------------------------
    test_cache_client()
    test_rotation()
