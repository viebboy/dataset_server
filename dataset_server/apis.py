"""
apis.py: interfaces for dataset server
--------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

This is part of the dataset_server project

License
-------
Apache License 2.0


"""

import dill
import tempfile
import time
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
from multiprocessing import shared_memory as SM
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


def extract_data_from_workers(workers, data_queue, max_queue_size, shuffle, close_event):
    indices = list(range(len(workers)))
    if shuffle:
        random.shuffle(indices)

    while True:
        if close_event.is_set():
            logger.info(f'receive signal to close the thread that pulls data from workers')
            return

        if data_queue.qsize() >= max_queue_size:
            time.sleep(0.001)
        else:
            minibatch = None
            while not close_event.is_set():
                for i in indices:
                    # if shuffle, then we get without waiting
                    response = workers[i].get(wait = not shuffle)
                    if response == 'epoch_end':
                        # this is the end of epoch, remove i from indices
                        indices.remove(i)
                    elif response is not None:
                        minibatch = response
                        break

                if len(indices) == 0:
                    indices = list(range(len(workers)))
                    if shuffle:
                        random.shuffle(indices)

                if minibatch is not None:
                    break
            data_queue.put(minibatch)


class Worker(CTX.Process):
    """
    Worker process that handles small part of the dataset loading
    (1) construct this child process
    (2) start it
    (3) check the status via .is_ready()
    (4) when the child process is ready, we can start getting minibatch via .get()
    """
    def __init__(self, name, **kwargs):
        super().__init__()
        self.verify_arguments(kwargs)
        self.name = name
        self.nb_minibatch = kwargs['nb_minibatch']
        self.cur_minibatch = 0

        # create pipe
        self._front_read_pipe, self._back_write_pipe = CTX.Pipe()
        self._back_read_pipe, self._front_write_pipe = CTX.Pipe()

        # create random name and shared memory from this name
        self.shared_memory = SM.SharedMemory(
            create=True,
            size=kwargs['max_minibatch_length']
        )
        kwargs['memory_name'] = self.shared_memory.name

        self.kwargs = kwargs
        self.is_closed = False
        self._is_ready = False

    def verify_arguments(self, kwargs):
        keys = [
            'loader_index',
            'max_queue_size',
            'dataset_module_file',
            'dataset_params_file',
            'batch_size',
            'shuffle',
            'nearby_shuffle',
            'max_minibatch_length',
            'nb_loader',
            'gpu_index',
        ]

        for key in keys:
            assert key in kwargs

    def read_pipe(self):
        return self._front_read_pipe

    def write_pipe(self):
        return self._front_write_pipe

    def run(self):
        import os
        if self.kwargs['gpu_index'] is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.kwargs['gpu_index'])

        from dataset_server.worker import DatasetLoader
        import asyncio

        # attach to shared memory
        shared_memory = SM.SharedMemory(name=self.kwargs['memory_name'])

        loader = DatasetLoader(
            shared_memory=shared_memory,
            read_pipe=self._back_read_pipe,
            write_pipe=self._back_write_pipe,
            dataset_module_file=self.kwargs['dataset_module_file'],
            dataset_params_file=self.kwargs['dataset_params_file'],
            nb_loader=self.kwargs['nb_loader'],
            loader_index=self.kwargs['loader_index'],
            batch_size=self.kwargs['batch_size'],
            max_queue_size=self.kwargs['max_queue_size'],
            shuffle=self.kwargs['shuffle'],
            nearby_shuffle=self.kwargs['nearby_shuffle'],
            name=self.name,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(loader.run())

    def process_message(self):
        status = self.read_pipe().recv()
        title = status['title']
        content = status['content']
        if title == 'minibatch_length':
            minibatch = dill.loads(self.shared_memory.buf[:content])
            self.cur_minibatch += 1
            self.write_pipe().send('can_send')
            return minibatch
        elif title == 'readiness':
            self.write_pipe().send('can_send')
            return True
        elif title == 'close_with_error':
            msg = (
                f'subprocess_error: the client process ({self.name}) has closed ',
                f'facing the following error: {content}'
            )
            # the subprocess closes by it own
            # we only need to handle from this side
            self.close_without_notice()
            raise RuntimeError(''.join(msg))
        else:
            # tell subprocess to close
            self.write_pipe().send('close')
            # then clean up
            self.close_with_notice()
            # and raise Exception
            raise RuntimeError(f'unknown message title: {title} from client ({self.name})')

    def get(self, wait: bool):
        """
        return None if there's no minibatch and wait=False
        return "epoch_end" if it is the end of this epoch
        return minibatch otherwise
        """
        if not self._is_ready:
            self.close_with_notice()
            raise RuntimeError('ChildProcess.get() should be called only when process is ready')

        if self.cur_minibatch == self.nb_minibatch:
            self.cur_minibatch = 0
            return 'epoch_end'

        if wait:
            return self.process_message()
        else:
            if self.read_pipe().poll():
                return self.process_message()
            else:
                return None

    def close_with_notice(self):
        if not self.is_closed:
            # perform clean up here
            self.shared_memory.close()
            self.shared_memory.unlink()
            # need to send noti to the subprocess
            self.write_pipe().send('close')
            self.write_pipe().close()
            self.is_closed = True

    def close_without_notice(self):
        if not self.is_closed:
            # perform clean up here
            self.shared_memory.close()
            self.shared_memory.unlink()
            # note that we only close the write end of the pipe
            # subprocess will handle the other end
            self.write_pipe().close()
            self.is_closed = True

    def is_ready(self):
        if self.read_pipe().poll():
            self._is_ready = self.process_message()
            return self._is_ready
        else:
            return False


class DataLoader:
    def __init__(
        self,
        dataset_class,
        dataset_params: dict,
        batch_size: int,
        nb_worker: int,
        max_minibatch_length=None,
        max_queue_size: int=20,
        prefetch_time=None,
        shuffle: bool=False,
        device=None,
        pin_memory=False,
        gpu_indices=None,
        nearby_shuffle: int=0,
        cache_setting=None,
        rotation_setting=None,
        use_threading=False,
        collate_fn=None,
    ):

        self.workers = []
        self.dataset_params_file = None

        dataset_len, max_minibatch_length = self.check_dataset(
            dataset_class,
            dataset_params,
            max_minibatch_length,
            batch_size,
        )

        if gpu_indices is not None:
            assert len(gpu_indices) == nb_worker

        # check the cache setting
        self.check_cache_setting(cache_setting)

        # assign batch size then check rotation setting
        rotation_setting = self.check_rotation_setting(rotation_setting)
        worker_len = self.compute_worker_length(
            dataset_len,
            nb_worker,
            batch_size,
            rotation_setting['rotation']
        )

        # start the subprocesses
        logger.info('starting subprocesses for data loading, this will take a while')

        self.start_workers(
            dataset_class=dataset_class,
            dataset_params=dataset_params,
            batch_size=batch_size,
            nb_worker=nb_worker,
            worker_len=worker_len,
            max_queue_size=max_queue_size,
            max_minibatch_length=max_minibatch_length,
            shuffle=shuffle,
            gpu_indices=gpu_indices,
            nearby_shuffle=nearby_shuffle,
            cache_setting=cache_setting,
            rotation_setting=rotation_setting,
            collate_fn=collate_fn,
        )

        self.batch_size = batch_size
        self.nb_worker = nb_worker
        self.shuffle = shuffle
        self.device = device
        self.pin_memory = pin_memory
        self.dataset_len = dataset_len
        self.worker_len = worker_len
        self.minibatch_count = 0
        self.latency_counter = None
        self.indices_to_process = list(range(nb_worker))
        self.use_threading = use_threading

        if use_threading:
            # now start a thread to pull from workers
            self.close_event = threading.Event()
            self.data_queue = Queue()
            self.pull_data_thread = threading.Thread(
                target=extract_data_from_workers,
                args=(
                    self.workers,
                    self.data_queue,
                    max_queue_size,
                    shuffle,
                    self.close_event
                )
            )
            self.pull_data_thread.start()

        if prefetch_time is not None and prefetch_time > 0:
            logger.info(f'waiting {prefetch_time} seconds to prefetch data')
            time.sleep(prefetch_time)

    def compute_worker_length(self, dataset_len, nb_worker, batch_size, rotation):
        sample_per_worker = int(np.ceil(dataset_len / nb_worker))
        worker_len = []
        for i in range(nb_worker):
            start_idx = i * sample_per_worker
            stop_idx = min(dataset_len, (i + 1) * sample_per_worker)
            nb_minibatch = rotation * int(np.ceil((stop_idx - start_idx) / batch_size))
            worker_len.append(nb_minibatch)

        return worker_len

    def check_dataset(self, dataset_class, dataset_params, max_minibatch_length, batch_size):
        try:
            # try to construct dataset
            dataset_tmp = dataset_class(**dataset_params)
            nb_sample = len(dataset_tmp)
            if max_minibatch_length is None:
                sample_len = len(dill.dumps(dataset_tmp[0]))
                max_minibatch_length = int(sample_len * batch_size * 1.1)
            del dataset_tmp
            return nb_sample, max_minibatch_length

        except Exception as error:
            logger.warning('failed to construct the dataset with the following error')
            raise error

    def check_cache_setting(self, cache_setting):
        if cache_setting is not None:
            # cache side must be server or client
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


    def check_rotation_setting(self, rotation_setting):
        if rotation_setting is None:
            rotation_setting = {
                'rotation': 1,
                'medium': 'memory',
            }
        else:
            assert 'rotation' in rotation_setting
            assert 'medium' in rotation_setting
            assert rotation_setting['medium'] in ['memory', 'disk']

            if rotation_setting['medium'] == 'disk':
                assert 'prefix' in rotation_setting
                if not os.path.exists(os.path.dirname(rotation_setting['prefix'])):
                    prefix = rotation_setting['prefix']
                    raise RuntimeError('The directory of file_prefix ({prefix}) does not exist')

                if 'size' not in rotation_setting:
                    msg = (
                        'Rotating on disk require "size", ',
                        'which is the number of samples that will be written to disk temporarily'
                    )
                    raise RuntimeError(''.join(msg))

            if rotation_setting['rotation'] < 2:
                msg = (
                    'the number of rotations should be at least 2; ',
                    f'received {rotation_setting["rotation"]}'
                )
                raise RuntimeError(''.join(msg))

        if rotation_setting['rotation'] > 1:
            logger.warning(f'rotation is ON; the size of dataset will increase {rotation_setting["rotation"]} times')
            logger.warning('samples are inherently shuffled when rotation is ON')

        return rotation_setting

    def start_workers(
        self,
        dataset_class,
        dataset_params,
        batch_size,
        nb_worker,
        worker_len,
        max_queue_size,
        max_minibatch_length,
        shuffle,
        gpu_indices,
        nearby_shuffle,
        cache_setting,
        rotation_setting,
        collate_fn,
    ):
        class_name = dataset_class.__name__
        dataset_module_file = inspect.getfile(dataset_class)

        # dump dataset-related data to a random file
        dataset_params_file = get_random_file(length=32)
        # mainify dataset params:

        with open(dataset_params_file, 'wb') as fid:
            dill.dump(
                {
                    'class_name': class_name,
                    'params': dataset_params,
                    'cache_setting': cache_setting,
                    'rotation_setting': rotation_setting,
                    'collate_fn': collate_fn,
                },
                fid,
                recurse=True
            )

        self.dataset_params_file = dataset_params_file

        # now create the workers
        self.workers = []
        for i in range(nb_worker):
            name = f'Worker_{i}'
            params = {
                'loader_index': i,
                'max_queue_size': max_queue_size,
                'dataset_module_file': dataset_module_file,
                'dataset_params_file': dataset_params_file,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'nearby_shuffle': nearby_shuffle,
                'max_minibatch_length': max_minibatch_length,
                'nb_loader': nb_worker,
                'nb_minibatch': worker_len[i],
                'gpu_index': gpu_indices[i] if gpu_indices is not None else None,
            }
            worker = Worker(name, **params)
            worker.start()
            self.workers.append(worker)
            time.sleep(5)

        # now check if workers are ready
        indices = list(range(nb_worker))
        while True:
            for i in indices:
                try:
                    is_ready = self.workers[i].is_ready()
                    if is_ready:
                        logger.info(f'{self.workers[i].name} is ready')
                        indices.remove(i)
                except RuntimeError as err:
                    if str(err).startswith('subprocess_error'):
                        self.close_without_notice()
                    else:
                        self.close()
                    raise err

            if len(indices) == 0:
                break
            else:
                time.sleep(0.1)

    def close(self):
        if self.use_threading:
            self.close_event.set()

        # send close signal to the worker
        for worker in self.workers:
            try:
                worker.close_with_notice()
            except Exception as error:
                logger.warning(f'faced {error} when closing worker {worker.name}')

        # cleanup the dataset file and dataset params
        if self.dataset_params_file is not None and os.path.exists(self.dataset_params_file):
            os.remove(self.dataset_params_file)

        for worker in self.workers:
            worker.join()

        if self.use_threading:
            self.pull_data_thread.join()

        logger.info('complete cleaning up and closing the DataLoader instance')

    def close_without_notice(self):
        if self.use_threading:
            self.close_event.set()

        # send close signal to the worker
        for worker in self.workers:
            try:
                worker.close_without_notice()
            except Exception as error:
                logger.warning(f'faced {error} when closing worker {worker.name}')

        # cleanup the dataset file and dataset params
        if self.dataset_params_file is not None and os.path.exists(self.dataset_params_file):
            os.remove(self.dataset_params_file)

        for worker in self.workers:
            worker.join()

        if self.use_threading:
            self.pull_data_thread.join()

        logger.info('complete cleaning up and closing the DataLoader instance')


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return sum(self.worker_len)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.use_threading:
                return self.get_minibatch_with_threading()
            else:
                return self.get_minibatch_without_threading()
        except StopIteration:
            raise StopIteration
        except Exception as error:
            self.close()
            raise error

    def get_minibatch_without_threading(self):
        if self.minibatch_count >= len(self):
            self.minibatch_count = 0
            raise StopIteration

        minibatch = None
        while minibatch is None:
            for i in self.indices_to_process:
                try:
                    response = self.workers[i].get(wait = not self.shuffle)
                    if response == 'epoch_end':
                        self.indices_to_process.remove(i)
                    elif response is not None:
                        minibatch = response
                        break
                except Exception as error:
                    if str(error).startswith('subprocess_error'):
                        self.close_without_notice()
                    else:
                        self.close()
                    raise error

            if len(self.indices_to_process) == 0:
                self.indices_to_process = list(range(self.nb_worker))
                self.minibatch_count = 0
                if self.shuffle:
                    random.shuffle(self.indices_to_process)

        if len(self.indices_to_process) == 0:
            self.indices_to_process = list(range(self.nb_worker))
            if self.shuffle:
                random.shuffle(self.indices_to_process)

        # increase the minibatch counter
        self.minibatch_count += 1

        if self.pin_memory:
            minibatch = pin_data_memory(minibatch)

        if self.device is not None:
            minibatch = move_data_to_device(minibatch, device)

        return minibatch

    def get_minibatch_with_threading(self):
        if self.minibatch_count >= len(self):
            self.minibatch_count = 0
            raise StopIteration

        while True:
            try:
                minibatch = self.data_queue.get(block=False)
                break
            except queue.Empty:
                time.sleep(0.0001)
                pass

        # increase the minibatch counter
        self.minibatch_count += 1

        if self.pin_memory:
            minibatch = pin_data_memory(minibatch)

        if self.device is not None:
            minibatch = move_data_to_device(minibatch, device)

        return minibatch
