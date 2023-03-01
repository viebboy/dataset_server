"""
apis.py: interfaces for dataset server
--------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

import subprocess
import dill
import tempfile
import time
import socket
import sys
import string
import random
import os
import numpy as np
import queue
from queue import Queue
import threading
from loguru import logger
import inspect
import multiprocessing as MP
from dataset_server.common import (
    BinaryBlob,
    shuffle_indices,
    Property,
)


CTX = MP.get_context('spawn')


def get_random_file(length):
    assert 0 < length < 256
    alphabet = list(string.ascii_lowercase)
    random_name = [random.choice(alphabet) for _ in range(length)]
    random_name = os.path.join(tempfile.gettempdir(), ''.join(random_name))
    return random_name

def move_data_to_device(data, device):
    if isinstance(data, (tuple, list)):
        return [move_data_to_device(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        raise RuntimeError(f'must be a torch tensor to move to device. Got {type(data)}')

def pin_data_memory(x):
    if isinstance(x, (tuple, list)):
        return [pin_data_memory(item) for item in x]
    else:
        return x.pin_memory()


class ClientProcess(CTX.Process):
    """
    client process that handles 1 connection to 1 server
    """
    def __init__(self, data_queue, event_queue, **kwargs):
        super().__init__()
        self.verify_arguments(kwargs)
        self.kwargs = kwargs
        self.data_queue = data_queue
        self.event_queue = event_queue

    def verify_arguments(self, kwargs):
        keys = [
            'client_index',
            'port',
            'max_queue_size',
            'packet_size',
            'retry_interval',
            'nb_retry',
            'cache',
        ]
        for key in keys:
            assert key in kwargs

    def run(self):
        from dataset_server.client import DatasetClient
        import asyncio
        client = DatasetClient(self.data_queue, self.event_queue, **self.kwargs)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.run())


class AsyncDataLoader:
    def __init__(
        self,
        dataset_class,
        dataset_params: dict,
        batch_size: int,
        nb_servers: int,
        start_port: int=11111,
        max_queue_size: int=20,
        shuffle: bool=False,
        device=None,
        pin_memory=False,
        packet_size=125000,
        wait_time=10,
        retry_interval=10,
        nb_retry=10,
        gpu_indices=None,
        nearby_shuffle: int=0,
        cache_setting=None,
        client_rotation_setting=None,
        server_rotation_setting=None,
    ):
        try:
            dataset_tmp = dataset_class(**dataset_params)
            del dataset_tmp
        except Exception as error:
            logger.warning('failed to construct the dataset with the following error')
            raise error

        if gpu_indices is not None:
            assert len(gpu_indices) == nb_servers

        # check the cache setting
        self.check_cache_setting(cache_setting)

        # assign batch size then check rotation setting
        self.batch_size = batch_size
        self.rotation = self.check_rotation_setting(client_rotation_setting, 'client', max_queue_size)
        self.check_rotation_setting(server_rotation_setting, 'server', max_queue_size)


        # start the servers
        logger.info('starting services, this will take a while')
        self.status_files = self.start_servers(
            dataset_class=dataset_class,
            dataset_params=dataset_params,
            batch_size=batch_size,
            nb_servers=nb_servers,
            start_port=start_port,
            max_queue_size=max_queue_size,
            shuffle=shuffle,
            packet_size=packet_size,
            gpu_indices=gpu_indices,
            nearby_shuffle=nearby_shuffle,
            cache_setting=cache_setting,
            rotation_setting=server_rotation_setting,
        )
        self.wait_for_servers()

        self.is_closed = False
        self.is_client_available = False

        # start clients
        client_params = {
            'max_queue_size': max_queue_size,
            'retry_interval': retry_interval,
            'nearby_shuffle': nearby_shuffle,
            'cache': self.cache,
            'packet_size': packet_size,
            'nb_retry': nb_retry,
        }
        self.start_clients(client_params)
        self.is_client_available = True

        # counter to track minibatch
        self.minibatch_count = 0

        # event check frequency
        self.event_check_counter = 100

        # start the thread to reconstruct data and put them into a queue
        self.device = device
        self.pin_memory = pin_memory

        # wait a bit to generate some samples before returning
        logger.info(f'waiting {wait_time} seconds to preload some samples')
        time.sleep(wait_time)

    def check_cache_setting(self, cache_setting):
        self.cache = None
        if cache_setting is not None:
            # cache side must be server or client
            assert 'side' in cache_setting
            assert cache_setting['side'] in ['server', 'client']
            assert 'prefix' in cache_setting
            if not os.path.exists(os.path.dirname(cache_setting['prefix'])):
                prefix = cache_setting['prefix']
                raise RuntimeError('The directory of cache_prefix ({prefix}) does not exist')
            assert 'update_frequency' in cache_setting
            if cache_setting['update_frequency'] < 2:
                msg = (
                    "there's no point in caching if update_frequency is less than 2; ",
                    f"received update_frequency={update_frequency}"
                )
                raise RuntimeError(''.join(msg))

            # now if caching on client side, create a property space to hold
            # all of properties
            if cache_setting['side'] == 'client':
                self.cache = Property()
                self.cache.side = cache_setting['side']
                self.cache.prefix = cache_setting['prefix']
                self.cache.update_frequency = cache_setting['update_frequency']


    def check_rotation_setting(self, rotation_setting, side, max_queue_size):
        rotation = Property()
        if rotation_setting is None:
            rotation.max_rotation = 1
            rotation.min_size = 1
            rotation.max_size = max_queue_size
            rotation.medium = 'memory'
        else:
            assert 'rotation' in rotation_setting
            assert 'min_rotation_size' in rotation_setting
            assert 'max_rotation_size' in rotation_setting
            rotation.max_rotation = rotation_setting['rotation']
            rotation.min_size = rotation_setting['min_rotation_size']
            rotation.max_size = rotation_setting['max_rotation_size']

            assert 'medium' in rotation_setting
            assert rotation_setting['medium'] in ['memory', 'disk']
            rotation.medium = rotation_setting['medium']

            if rotation.medium == 'disk':
                assert 'prefix' in rotation_setting
                if not os.path.exists(os.path.dirname(rotation_setting['prefix'])):
                    prefix = rotation_setting['prefix']
                    raise RuntimeError('The directory of file_prefix ({prefix}) does not exist')
                rotation.prefix = rotation_setting['prefix']

            if side == 'client' and rotation.medium == 'disk':
                msg = (
                    'rotation on disk on client side is not supported; ',
                    'consider using memory as rotation medium for client_rotation_setting',
                )
                raise RuntimeError(''.join(msg))

            if rotation.max_rotation < 2:
                msg = (
                    'the number of rotations should be at least 2; ',
                    f'received {rotation.max_rotation}'
                )
                raise RuntimeError(''.join(msg))

            assert rotation.min_size >= 1,\
                'min_rotation_size must be at least 1'

            assert rotation.max_size >= 2,\
                'max_rotation_size must be at least 2'

            if rotation.max_size <= rotation.min_size:
                msg = (
                    'max_rotation_size must be larger than min_rotation_size; ',
                    f'received max_rotation_size={rotation.max_size}, ',
                    f'min_rotation_size={rotation.min_size}'
                )
                raise RuntimeError(''.join(msg))

            invalid_min_size = (
                rotation.max_rotation > 1 and
                side == 'server' and
                rotation.medium == 'memory' and
                rotation.min_size <= self.batch_size
            )
            if invalid_min_size:
                msg = (
                    'when rotation on memory on server side is specified, ',
                    f'min_rotation_size ({rotation.min_size}) must be larger ',
                    f'than batch_size ({self.batch_size}); ',
                    'this is to ensure that a sample is not repeated many times in the same minibatch',
                )
                raise RuntimeError(''.join(msg))

            invalid_max_size = (
                rotation.max_rotation > 1 and
                side == 'server' and
                rotation.medium == 'disk' and
                rotation.max_size < self.batch_size
            )
            if invalid_max_size:
                msg = (
                    'when rotation on disk on server side is specified, ',
                    f'max_rotation_size ({rotation.max_size}) must be at least ',
                    f'batch_size ({self.batch_size}); ',
                    'this is to ensure that enough samples are written to file for rotation',
                )
                raise RuntimeError(''.join(msg))

        return rotation


    def wait_for_servers(self):
        logger.info('waiting for servers to load dataset...')
        while True:
            ready = True
            for file in self.status_files:
                if not os.path.exists(file):
                    ready = False
                    break
            if not ready:
                time.sleep(1)
            else:
                break

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def start_clients(self, params):
        """
        start client processes
        """
        self.clients = []
        self.client_indices = []
        self.data_queues = []
        self.event_queues = []


        for index, port in enumerate(self.ports):
            logger.info(f'starting DatasetClient number {index} at port: {port}')
            # construct parameters for client process
            client_params = {key: value for key, value in params.items()}
            client_params['client_index'] = index
            client_params['port'] = port

            data_queue = CTX.Queue()
            event_queue = CTX.Queue()
            self.data_queues.append(data_queue)
            self.event_queues.append(event_queue)
            self.clients.append(ClientProcess(data_queue, event_queue, **client_params))
            self.client_indices.append(index)

        self.nb_client = len(self.client_indices)

        # start the clients
        for client in self.clients:
            client.start()

        # now loop through event_queues to check if all processes are ready
        logger.info('waiting for client processes to be ready...')
        self.total_minibatch = 0
        status = [None for _ in range(self.nb_client)]
        while True:
            count = 0
            for i in range(self.nb_client):
                if status[i] is None:
                    try:
                        response = self.event_queues[i].get(block=False)
                        if response['status'] == 'ready':
                            status[i] = 1
                            count += 1
                            self.total_minibatch += response['total_minibatch']
                        elif response['status'] == 'failed':
                            logger.warning('one of the client processes failed')
                            logger.warning('force closing now...')
                            self.force_close('one of the client processes failed')
                    except queue.Empty:
                        time.sleep(1)
                else:
                    count += 1
            if count == self.nb_client:
                break
            else:
                time.sleep(1)
        logger.info('client processes are ready')


    def close(self):
        if not self.is_closed:
            logger.info('closing the AsyncDataLoader instance')
            # close the thread running DatasetClient
            if self.is_client_available:
                # signal the client process to terminate
                for i in range(self.nb_client):
                    self.event_queues[i].put('close')

            for client in self.clients:
                client.join()

            # close the dataset servers
            for server in self.servers:
                server.kill()

            # delete status files
            for file in self.status_files:
                if os.path.exists(file):
                    os.remove(file)

            self.is_closed = True
            self.is_client_available = False

    def force_close(self, message):
        for server in self.servers:
            server.kill()

        for client in self.clients:
            client.kill()

        raise RuntimeError(message)

    def start_servers(
        self,
        dataset_class,
        dataset_params: dict,
        batch_size: int,
        nb_servers: int,
        start_port: int,
        max_queue_size: int,
        shuffle: bool,
        packet_size: int,
        gpu_indices: list,
        nearby_shuffle: bool,
        cache_setting: dict,
        rotation_setting: dict,
    ):
        """
        start the dataset servers
        """

        class_name = dataset_class.__name__
        dataset_module_file = inspect.getfile(dataset_class)
        logger.info(f'dataset_module file: {dataset_module_file}')
        logger.info(f'dataset_module name: {class_name}')

        # dump dataset-related data to a random file
        dataset_params_file = get_random_file(length=32)
        # mainify dataset params:
        #mainify(dataset_params)

        with open(dataset_params_file, 'wb') as fid:
            dill.dump(
                {
                    'class_name': class_name,
                    'params': dataset_params,
                    'cache_setting': cache_setting,
                    'rotation_setting': rotation_setting,
                },
                fid,
                recurse=True
            )

        # create random files to write status after server is available
        status_files = [get_random_file(32) for _ in range(nb_servers)]

        # start the server
        self.servers = []
        self.ports = []
        for server_idx in range(nb_servers):
            logger.info(f'starting dataset server {server_idx +1}')
            all_env_var = os.environ.copy()
            if gpu_indices is not None:
                indices = gpu_indices[server_idx]
                if isinstance(indices, int):
                    indices = [indices,]
                env_var = ','.join([str(v) for v in indices])
                #TODO: check the case when script is run with CUDA_VISIBLE_DEVICES
                all_env_var['CUDA_VISIBLE_DEVICES'] = env_var

            process = subprocess.Popen(
                [
                    'serve-dataset',
                    '--port',
                    str(start_port),
                    '--dataset-file',
                    dataset_module_file,
                    '--dataset-parameter-file',
                    dataset_params_file,
                    '--nb-server',
                    str(nb_servers),
                    '--server-index',
                    str(server_idx),
                    '--batch-size',
                    str(batch_size),
                    '--max-queue-size',
                    str(max_queue_size),
                    '--shuffle',
                    str(shuffle),
                    '--packet-size',
                    str(packet_size),
                    '--status-file',
                    status_files[server_idx],
                    '--nearby-shuffle',
                    str(nearby_shuffle),
                ],
                env=all_env_var,
            )
            time.sleep(2)
            self.servers.append(process)
            self.ports.append(start_port)
            start_port += 1

        return status_files

    def __len__(self):
        return max(1, self.rotation.max_rotation) * self.total_minibatch

    def __iter__(self):
        return self

    def __next__(self):
        if self.minibatch_count >= max(1, self.rotation.max_rotation) * self.total_minibatch:
            self.minibatch_count = 0
            raise StopIteration

        if self.event_check_counter > 0:
            self.event_check_counter -= 1
        else:
            self.event_check_counter = 100
            for i in range(self.nb_client):
                if not self.event_queues[i].empty():
                    try:
                        event = self.event_queues[i].get(block=False)
                        if event['status'] == 'failed':
                            logger.warning('one of the client processes failed')
                            logger.warning('force closing now...')
                            self.force_close('one of the client processes failed')
                        else:
                            raise RuntimeError(f'receive unknown event from client process: {event}')
                    except queue.Empty:
                        pass

        while True:
            # read from data_queue
            minibatch = self.data_queues[self.client_indices[0]].get()
            if minibatch is None:
                # this is the end of epoch from this client index
                # remove this client index from the list
                self.client_indices.pop(0)
                if len(self.client_indices) == 0:
                    self.client_indices = list(range(self.nb_client))
            else:
                self.client_indices.append(self.client_indices.pop(0))
                break

        # increase the minibatch counter
        self.minibatch_count += 1

        if self.pin_memory:
            minibatch = pin_data_memory(minibatch)

        if self.device is not None:
            minibatch = move_data_to_device(minibatch, device)

        return minibatch

    def get_sample_without_rotation(self):
        while True:
            for i in self.client_indices:
                try:
                    minibatch = self.data_queues[i].get(block=False)
                except queue.Empty:
                    pass

class AsyncDataset(AsyncDataLoader):
    def __init__(
        self,
        dataset_class,
        dataset_params: dict,
        nb_servers: int,
        start_port: int=11111,
        max_queue_size: int=20,
        shuffle: bool=False,
        device=None,
        pin_memory=False,
        packet_size=125000,
        wait_time=10,
        client_wait_time=10,
        nb_retry=10,
        gpu_indices=None,
        qt_threading=False,
        nearby_shuffle: int=0,
        cache_setting=None,
        rotation_setting=None,
    ):
        super().__init__(
            dataset_class=dataset_class,
            dataset_params=dataset_params,
            nb_servers=nb_servers,
            start_port=start_port,
            batch_size=1,
            max_queue_size=max_queue_size,
            shuffle=shuffle,
            device=device,
            pin_memory=pin_memory,
            packet_size=packet_size,
            wait_time=wait_time,
            client_wait_time=client_wait_time,
            nb_retry=nb_retry,
            gpu_indices=gpu_indices,
            qt_threading=qt_threading,
            nearby_shuffle=nearby_shuffle,
            cache_setting=cache_setting,
            rotation_setting=rotation_setting,
        )
