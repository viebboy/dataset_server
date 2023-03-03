from torchvision.datasets import CIFAR10
from dataset_server import DataLoader
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
        return x, i

def test_default(use_threading):
    logger.info(f'test default setting')
    batch_size = 64
    dataloader = DataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=batch_size,
        nb_worker=1,
        max_queue_size=128,
        nearby_shuffle=100,
        shuffle=False,
        use_threading=use_threading,
    )
    print(f'data loader length: {len(dataloader)}')

    for _ in range(1):
        start = 0
        for idx, samples in tqdm(enumerate(dataloader)):
            stop = start + samples[1].size
            true_lb = np.arange(start, stop).flatten()
            generated_lb = samples[1].flatten()
            np.testing.assert_equal(generated_lb, true_lb)
            start = stop

    dataloader.close()
    logger.info('complete testing default setting')


def test_cache(use_threading):
    logger.info(f'test caching setting in server side')
    cache_setting = {
        'prefix': './data/train_cache',
        'update_frequency': 3,
    }
    dataloader = DataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_worker=2,
        max_queue_size=128,
        nearby_shuffle=100,
        shuffle=True,
        cache_setting=cache_setting,
        use_threading=use_threading,
    )
    print(f'data loader length: {len(dataloader)}')

    start = time.time()
    for _ in range(1):
        for samples in tqdm(dataloader):
            time.sleep(0.01)

    stop = time.time()
    dataloader.close()
    logger.info('complete testing caching')

def test_rotation_on_memory(use_threading):
    logger.info(f'test rotation on memory')
    rotation_setting = {
        'rotation': 3,
        'medium': 'memory',
    }
    dataloader = DataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_worker=2,
        max_queue_size=100,
        nearby_shuffle=100,
        rotation_setting=rotation_setting,
        use_threading=use_threading,
    )

    start = time.time()
    for _ in range(1):
        for samples in tqdm(dataloader):
            time.sleep(0.01)

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on memory')

def test_rotation_on_disk(use_threading):
    logger.info(f'test rotation on disk')
    rotation_setting = {
        'rotation': 5,
        'size': 10_000,
        'prefix': './data/client_rotation',
        'medium': 'disk',
    }
    dataloader = DataLoader(
        dataset_class=Dataset,
        dataset_params={'train': True},
        batch_size=64,
        nb_worker=1,
        max_queue_size=10,
        shuffle=True,
        nearby_shuffle=100,
        rotation_setting=rotation_setting,
        use_threading=use_threading,
    )

    start = time.time()
    for _ in range(1):
        for samples in tqdm(dataloader):
            time.sleep(0.01)

    stop = time.time()
    dataloader.close()
    logger.info('complete testing rotation on client on disk')



if __name__ == '__main__':
    #---------------------------------------
    #test_cache_server()
    #test_cache_client()
    #test_rotation_on_memory()
    #test_rotation_on_client_on_disk()
    #test_rotation_on_server_on_memory()
    test_rotation_on_disk(True)
    #test_rotation_on_disk(False)
    #test_rotation_on_memory(True)
    #test_default(False)
    #test_default(False)
    #test_rotation_on_disk(False)
