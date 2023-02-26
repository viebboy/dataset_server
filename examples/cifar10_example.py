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

def test_default(qt_threading):
    logger.info(f'test default setting with qt_threading={qt_threading}')
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        qt_threading=qt_threading,
    )
    print(f'data loader length: {len(dataloader)}')

    start = time.time()
    for samples in tqdm(dataloader):
        pass
    stop = time.time()
    dataloader.close()
    logger.info('complete testing the default config')

def test_cache_client():
    logger.info(f'test caching setting in client side with')
    cache_setting = {
        'side': 'client',
        'prefix': './data/train_cache',
        'update_frequency': 3,
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        nearby_shuffle=100,
        cache_setting=cache_setting,
    )
    print(f'data loader length: {len(dataloader)}')

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing caching on client side')

def test_cache_server():
    logger.info(f'test caching setting in server side')
    cache_setting = {
        'side': 'server',
        'prefix': './data/train_cache',
        'update_frequency': np.inf,
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=128,
        nearby_shuffle=100,
        cache_setting=cache_setting,
    )
    print(f'data loader length: {len(dataloader)}')

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing caching on server side')

def test_rotation_on_client_on_memory():
    logger.info(f'test rotation on client on memory')
    client_rotation_setting = {
        'rotation': 3,
        'min_rotation_size': 10,
        'max_rotation_size': 100,
        'medium': 'memory',
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=10,
        nearby_shuffle=100,
        client_rotation_setting=client_rotation_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on client on memory')

def test_rotation_on_server_on_memory():
    logger.info(f'test rotation on server on memory')
    server_rotation_setting = {
        'rotation': 3,
        'min_rotation_size': 65,
        'max_rotation_size': 1000,
        'medium': 'memory',
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=10,
        nearby_shuffle=100,
        server_rotation_setting=server_rotation_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on client on memory')

def test_rotation_on_client_on_disk():
    logger.info(f'test rotation on client on disk')
    client_rotation_setting = {
        'rotation': 3,
        'min_rotation_size': 10,
        'max_rotation_size': 100,
        'prefix': './data/client_rotation',
        'medium': 'disk',
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=10,
        nearby_shuffle=100,
        client_rotation_setting=client_rotation_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on client on disk')

def test_rotation_on_server_on_disk():
    logger.info(f'test rotation on server on disk')
    server_rotation_setting = {
        'rotation': 3,
        'min_rotation_size': 65,
        'max_rotation_size': 6400,
        'prefix': './data/server_rotation',
        'medium': 'disk',
    }
    dataloader = AsyncDataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_servers=1,
        start_port=50000,
        max_queue_size=10,
        nearby_shuffle=100,
        server_rotation_setting=server_rotation_setting,
    )

    start = time.time()
    for _ in range(6):
        for samples in tqdm(dataloader):
            pass

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on client on disk')




if __name__ == '__main__':
    #---------------------------------------
    #test_cache_server()
    #test_cache_client()
    #test_rotation_on_client_on_memory()
    #test_rotation_on_client_on_disk()
    #test_rotation_on_server_on_memory()
    test_rotation_on_server_on_disk()
    #test_rotation_on_disk(False)
