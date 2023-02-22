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
from queue import Queue
import threading
from loguru import logger
import inspect
from dataset_server.common import BinaryBlob, shuffle_indices

# size of the header
INTERCOM_HEADER_LEN = 4

try:
    import __builtin__ as B
except ImportError:
    import builtins as B

BUILTIN_TYPES = []
for t in B.__dict__.values():
    if isinstance(t, type) and t is not object and t is not type:
        BUILTIN_TYPES.append(t)


def is_builtin_type(obj):
    output = False
    for t in BUILTIN_TYPES:
        if isinstance(obj, t):
            output = True
            break
    return output


def mainify(obj):
    """
    If obj is not defined in __main__ then redefine it in
    main so that dill will serialize the definition along with the object
    Taken from:
    https://oegedijk.github.io/blog/pickle/dill/python/2020/11/10/serializing-dill-references.html

    Modified so that mainify is only applied to custom Class
    """

    # if obj is a class and is not a builtin type
    if inspect.isclass(obj) and not is_builtin_type(obj):
        if obj.__module__ != "__main__":
            import __main__
            s = inspect.getsource(obj)
            co = compile(s, '<string>', 'exec')
            exec(co, __main__.__dict__)
            logger.debug(f'mainifying class : {obj}')
        else:
            logger.debug(f'get a class : {obj} but scope is in main')
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            mainify(item)
    elif isinstance(obj, dict):
        for _, value in obj.items():
            mainify(value)


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

def rotate_data(
    data_queue,
    rotation_queue,
    max_rotation_size,
    close_event,
):
    while True:
        if close_event.is_set():
            logger.info(f'receive signal to close rotate data thread thread, closing now...')
            return

        r_size = rotation_queue.qsize()
        if r_size <= max_rotation_size:
            for _ in range(max_rotation_size + 1 - r_size):
                if not data_queue.empty():
                    minibatch = data_queue.get()
                    rotation_queue.put((1, minibatch))
                else:
                    break
        else:
            time.sleep(0.001)


def fetch_data(
    clients,
    data_queue,
    max_queue_size,
    close_event,
    pin_memory,
    cache_setting,
):
    # get the sample from the 1st client in the client list
    nb_client = len(clients)
    indices = list(range(nb_client))
    record = None
    current_epoch = 0
    minibatch_count = 0
    total_minibatch = sum([len(client) for client in clients])

    if cache_setting is not None and cache_setting['cache_side'] == 'client':
        # need caching
        cache_binary_file = cache_setting['cache_prefix'] + '.bin'
        cache_index_file = cache_setting['cache_prefix'] + '.idx'
        if os.path.exists(cache_index_file) and os.path.exists(cache_binary_file):
            # if rewrite, we delete the files
            if cache_setting['rewrite']:
                os.remove(cache_index_file)
                os.remove(cache_binary_file)
                record = BinaryBlob(cache_binary_file, cache_index_file, mode='w')
            else:
                record = BinaryBlob(cache_binary_file, cache_index_file, mode='r')
        else:
            record = BinaryBlob(cache_binary_file, cache_index_file, mode='w')

        cache_update_frequency = cache_setting['update_frequency']

    # handle the case when no device is specified
    while True:
        if close_event.is_set():
            logger.info(f'receive signal to close data fetcher thread')
            if record is not None:
                record.close()
            return

        # check if sample should be reconstructed from server or from cache
        if record is None:
            from_server = True
        else:
            if current_epoch == 0:
                # 1st epoch, if record in read mode
                # it means cache exists, just need to read from cache
                if record.mode() == 'read':
                    from_server = False
                else:
                    from_server = True
            else:
                if current_epoch % cache_update_frequency == 0:
                    from_server = True
                else:
                    from_server = False

        if data_queue.qsize() <= max_queue_size:
            try:
                if from_server:
                    # if reading from server
                    # check if 1st sample from the server and record in
                    # read mode, then we need to close the record and open
                    # again in write mode
                    if minibatch_count == 0 and record is not None and record.mode() == 'read':
                        # close
                        record.close()
                        # then open again for writing
                        record = BinaryBlob(
                            cache_binary_file,
                            cache_index_file,
                            mode='w',
                        )

                    leftover, minibatch = clients[indices[0]].get()
                    if record is not None:
                        record.write_index(minibatch_count, minibatch)

                    if pin_memory:
                        minibatch = pin_data_memory(minibatch)

                    data_queue.put(minibatch)
                    minibatch_count += 1

                    if leftover == 0:
                        # remove the client from the list
                        indices.pop(0)
                        if len(indices) == 0:
                            # if empty, reinitialize
                            indices = list(range(nb_client))
                            # then reset the counter
                            minibatch_count = 0
                            current_epoch += 1
                            if record is not None:
                                # if record is not None, we also need to close
                                # the record to finish the writing process
                                record.close()
                                # then open record again in read mode
                                record = BinaryBlob(
                                    cache_binary_file,
                                    cache_index_file,
                                    mode='r',
                                )
                    else:
                        # rotate the list
                        indices.append(indices.pop(0))
                else:
                    # if reading from cache file
                    minibatch = record.read_index(minibatch_count)

                    if pin_memory:
                        minibatch = pin_data_memory(minibatch)

                    data_queue.put(minibatch)
                    minibatch_count += 1

                    # reset counter
                    if minibatch_count == total_minibatch:
                        current_epoch += 1
                        minibatch_count = 0

            except RuntimeError as e:
                # put None to data queue
                # then put error message
                data_queue.put(None)
                data_queue.put(str(e))
                for client in clients:
                    client.close()
                raise RuntimeError(str(e))
        else:
            time.sleep(0.001)


class CloseEvent:
    def __init__(self, event_queue):
        self.event_queue = event_queue

    def is_set(self):
        return not self.event_queue.empty()

try:
    from PySide6.QtCore import QObject, QThread

    class DataFetcher(QObject):
        def __init__(
            self,
            clients,
            data_queue,
            max_queue_size,
            pin_memory,
            cache_setting
        ):
            super().__init__()
            self.clients = clients
            self.data_queue = data_queue
            self.max_queue_size = max_queue_size
            self.pin_memory = pin_memory
            self.close_signal = Queue()
            self.cache_setting = cache_setting

        def run(self):
            # get the sample from the 1st client in the client list
            fetch_data(
                clients=self.clients,
                data_queue=self.data_queue,
                max_queue_size=self.max_queue_size,
                close_event=CloseEvent(self.close_signal),
                pin_memory=self.pin_memory,
                cache_setting=self.cache_setting,
            )


    class DataRotator(QObject):
        def __init__(
            self,
            data_queue,
            rotation_queue,
            max_rotation_size,
        ):
            super().__init__()
            self.data_queue = data_queue
            self.rotation_queue = rotation_queue
            self.max_rotation_size = max_rotation_size
            self.close_signal = Queue()

        def run(self):
            # get the sample from the 1st client in the client list
            rotate_data(
                data_queue=self.data_queue,
                rotation_queue=self.rotation_queue,
                max_rotation_size=self.max_rotation_size,
                close_event=CloseEvent(self.close_signal),
            )


except ImportError:
    DataFetcher = None
    DataRotator = None


class DatasetClient:
    def __init__(self, index, port, packet_size: int, hostname='localhost', wait_time=10, nb_retry=10):
        self.index = index
        self.port = port
        self.hostname = hostname
        self.socket = None
        self.wait_time = wait_time
        self.nb_retry = nb_retry
        self.size = None
        self.packet_size = packet_size

        # initialize connection
        self.init_connection()

        # counter to track minibatch
        self.minibatch_count = 0

    def __len__(self):
        return self.size

    def get(self):
        try:
            if self.minibatch_count == self.size:
                self.minibatch_count = 0

            samples = self.read_socket()
            self.write_socket('ok')
            self.minibatch_count += 1
            return self.size - self.minibatch_count, samples

        except Exception as e:
            logger.warning('face the following exception')
            logger.warning(str(e))
            self.close()
            raise RuntimeError(str(e))

    def close(self):
        if self.socket is not None:
            logger.info(f'closing connection to {self.hostname} at port {self.port}')
            self.socket.close()
            self.socket = None

    def init_connection(self):
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            count = 0

            # try to connect
            success = False
            while True:
                try:
                    self.socket.connect((self.hostname, self.port))
                    logger.info(f'connected to server {self.hostname} at port {self.port}')
                    success = True
                    break
                except Exception as e:
                    time.sleep(self.wait_time)
                    count += 1
                    msg = (
                        f'failed to connect at the {count}-th attempt! ',
                        f'wating {self.wait_time} seconds before retrying'
                    )
                    logger.warning(''.join(msg))
                    if count >= self.nb_retry:
                        logger.warning(f'failed to connect after retrying {self.nb_retry} times')
                        logger.warning('terminating now!!!')
                        break

            # disable naggle algorithm
            #self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

            # now read the size of the dataset (number of minibatches)
            # the size is sent as 4 bytes
            if success:
                self.size = int.from_bytes(self.socket.recv(4), 'big')
                self.write_socket('ok')
            else:
                raise RuntimeError(f'failed to connect after retrying {self.nb_retry} times')

    def read_socket(self):
        """
        first read header (4 bytes), which contains the size of the object
        then read the object
        """
        # read the header
        header = self.socket.recv(INTERCOM_HEADER_LEN)
        # compute the object length
        obj_len = int.from_bytes(header, 'big')
        # then read the object length
        nb_byte_left = obj_len
        byte_rep = bytes()
        while nb_byte_left > 0:
            block_size = min(nb_byte_left, self.packet_size)
            bytes_ = self.socket.recv(block_size)
            if len(bytes_) < 1:
                sys.exit(2)
            byte_rep += bytes_
            nb_byte_left -= len(bytes_)

        samples = dill.loads(byte_rep)
        return samples

    def write_socket(self, message):
        # note that this implementation can only send small message
        # if long message then we need to divide into chunks
        if self.socket is not None:
            byte_rep = dill.dumps(message)
            payload = len(byte_rep).to_bytes(4, 'big') + byte_rep
            self.socket.sendall(payload)


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
        client_wait_time=10,
        nb_retry=10,
        gpu_indices=None,
        qt_threading=False,
        nearby_shuffle: int=0,
        cache_setting=None,
        rotation_setting=None,
    ):
        try:
            dataset_tmp = dataset_class(**dataset_params)
            del dataset_tmp
        except Exception as error:
            logger.warning('failed to construct the dataset with the following error')
            raise error

        if gpu_indices is not None:
            assert len(gpu_indices) == nb_servers

        if cache_setting is not None:
            # cache side must be server or client
            assert 'cache_side' in cache_setting
            assert cache_setting['cache_side'] in ['server', 'client']
            assert 'cache_prefix' in cache_setting
            if not os.path.exists(os.path.dirname(cache_setting['cache_prefix'])):
                prefix = cache_setting['cache_prefix']
                raise RuntimeError('The directory of cache_prefix ({prefix}) does not exist')
            assert 'rewrite' in cache_setting
            assert 'update_frequency' in cache_setting
            assert cache_setting['update_frequency'] >= 2

        if rotation_setting is None:
            self.rotation = 1
            self.min_rotation_size = 1
            self.max_rotation_size = max_queue_size
        else:
            assert 'rotation' in rotation_setting
            assert 'min_rotation_size' in rotation_setting
            assert 'max_rotation_size' in rotation_setting

            assert rotation_setting['rotation'] >= 2
            self.rotation = rotation_setting['rotation']
            logger.warning(
                f'rotation option is ON. This means the dataset will be replicate {self.rotation} times'
            )
            assert rotation_setting['min_rotation_size'] >= 2,\
                'min_rotation_size must be at least 2'
            self.min_rotation_size = rotation_setting['min_rotation_size']
            assert rotation_setting['max_rotation_size'] >= 3,\
                'max_rotation_size must be at least 3'
            self.max_rotation_size = rotation_setting['max_rotation_size']

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
            cache_setting=cache_setting,
            nearby_shuffle=nearby_shuffle,
        )
        self.wait_for_servers()

        self.is_closed = False
        self.is_client_available = False

        if qt_threading:
            msg = (
                f'qt_threading option is True. To use threading from Qt framework, ',
                'PySide6 needs to be installed'
            )
            logger.warning(''.join(msg))

        if qt_threading:
            if DataFetcher is not None and DataRotator is not None:
                self.qt_threading = True
                logger.warning('found working installation of Pyside6, QThread is in used')
            else:
                logger.warning('cannot find working installation of Pyside6, use default python threading')
                self.qt_threading = False
        else:
            self.qt_threading = False

        # start clients
        self.start_clients(packet_size, client_wait_time, nb_retry)
        self.is_client_available = True

        # counter to track minibatch
        self.minibatch_count = 0

        # start the thread to reconstruct data and put them into a queue
        self.data_queue = Queue()
        self.rotation_queue = Queue()
        self.device = device

        if not self.qt_threading:
            # use python threading
            self.fetcher_close_event = threading.Event()
            self.data_thread = threading.Thread(
                target=fetch_data,
                args=(
                    self.clients,
                    self.data_queue,
                    max_queue_size,
                    self.fetcher_close_event,
                    pin_memory,
                    cache_setting,
                )
            )
            self.data_thread.start()

            # thread for data rotation
            self.rotator_close_event = threading.Event()
            self.rotator_thread = threading.Thread(
                target=rotate_data,
                args=(
                    self.data_queue,
                    self.rotation_queue,
                    self.max_rotation_size,
                    self.rotator_close_event,
                )
            )
            self.rotator_thread.start()

        else:
            # use qt threading
            self.data_thread = QThread()
            self.data_fetcher = DataFetcher(
                clients=self.clients,
                data_queue=self.data_queue,
                max_queue_size=max_queue_size,
                pin_memory=pin_memory,
                cache_setting=cache_setting,
            )
            self.data_fetcher.moveToThread(self.data_thread)
            self.data_thread.started.connect(self.data_fetcher.run)
            self.data_thread.start()

            self.rotator_thread = QThread()
            self.data_rotator = DataRotator(
                data_queue=self.data_queue,
                rotation_queue=self.rotation_queue,
                max_rotation_size=self.max_rotation_size,
            )
            self.data_rotator.moveToThread(self.rotator_thread)
            self.rotator_thread.started.connect(self.data_rotator.run)
            self.rotator_thread.start()


        # wait a bit to generate some samples before returning
        logger.info(f'waiting {wait_time} seconds to preload some samples')
        time.sleep(wait_time)

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

    def start_clients(self, packet_size, wait_time, nb_retry):
        """
        start the socket clients
        """
        self.clients = []
        self.sizes = []
        self.total_minibatch = 0
        self.client_indices = []

        for index, port in enumerate(self.ports):
            logger.info(f'starting DatasetClient number {index} at port: {port}')
            try:
                new_client = DatasetClient(
                    index=index,
                    port=port,
                    packet_size=packet_size,
                    wait_time=wait_time,
                    nb_retry=nb_retry,
                )
            except Exception as error:
                self.close()
                raise error

            self.clients.append(new_client)
            self.sizes.append(len(new_client))
            self.total_minibatch += len(new_client)
            self.client_indices.append(index)

        self.nb_client = len(self.sizes)

    def close(self):
        if not self.is_closed:
            logger.info('closing the AsyncDataLoader instance')
            # close the thread running DatasetClient
            if self.is_client_available:
                if not self.qt_threading:
                    self.fetcher_close_event.set()
                    self.rotator_close_event.set()
                    self.data_thread.join()
                    self.rotator_thread.join()
                else:
                    self.data_fetcher.close_signal.put(None)
                    self.data_rotator.close_signal.put(None)
                    self.data_thread.quit()
                    self.rotator_thread.wait()

                for client in self.clients:
                    client.close()

            # close the dataset servers
            for server in self.servers:
                server.kill()

            # delete status files
            for file in self.status_files:
                if os.path.exists(file):
                    os.remove(file)

            self.is_closed = True
            self.is_client_available = False

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
        cache_setting: dict,
        nearby_shuffle: bool,
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
        return max(1, self.rotation) * self.total_minibatch

    def __iter__(self):
        return self

    def __next__(self):
        if self.minibatch_count >= max(1, self.rotation) * self.total_minibatch:
            self.minibatch_count = 0
            raise StopIteration

        while True:
            if self.rotation > 1:
                # rotation is enabled
                if self.rotation_queue.qsize() >= self.min_rotation_size:
                    # has some minimum elements
                    readout = self.rotation_queue.get()
                    if readout is None:
                        error = self.rotation_queue.get()
                        self.close()
                        logger.warning(f'got error: {error}')
                        raise RuntimeError(error)

                    counter, minibatch = readout
                    if counter < self.rotation:
                        self.rotation_queue.put((counter+1, minibatch))
                    break
                else:
                    time.sleep(0.001)
            else:
                # no rotation, dont care about min rotation size
                if not self.rotation_queue.empty():
                    readout = self.rotation_queue.get()
                    if readout is None:
                        error = self.rotation_queue.get()
                        self.close()
                        logger.warning(f'got error: {error}')
                        raise RuntimeError(error)

                    counter, minibatch = readout
                    break
                else:
                    time.sleep(0.001)

        # increase the minibatch counter
        self.minibatch_count += 1

        if self.device is not None:
            minibatch = move_data_to_device(minibatch, device)

        return minibatch

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
