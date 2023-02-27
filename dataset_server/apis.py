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
from dataset_server.common import BinaryBlob, shuffle_indices, Property

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


def fetch_data(
    clients,
    data_queue,
    rotation_queue,
    max_queue_size,
    close_event,
    pin_memory,
    cache,
    rotation,
    nearby_shuffle,
):
    # get the sample from the 1st client in the client list
    nb_client = len(clients)
    client_indices = list(range(nb_client))
    record = None
    current_epoch = 0
    minibatch_count = 0
    total_minibatch = sum([len(client) for client in clients])

    # rotation counter
    rotation.read_indices = None
    rotation.current_round = 0
    rotation.write_idx = 0
    rotation.ignored_indices = []

    # handle cache setting
    if cache is not None and cache.side == 'client':
        # need caching
        cache_binary_file = cache.prefix + '.bin'
        cache_index_file = cache.prefix + '.idx'
        cache.record = BinaryBlob(cache_binary_file, cache_index_file, mode='w')
    else:
        cache = None

    # handle rotation setting
    if rotation.medium == 'disk':
        part_a = BinaryBlob(
            binary_file=rotation.prefix + 'A.bin',
            index_file=rotation.prefix + 'A.idx',
            mode='w',
        )
        part_b = BinaryBlob(
            binary_file=rotation.prefix + 'B.bin',
            index_file=rotation.prefix + 'B.idx',
            mode='w',
        )
        rotation.records = [part_a, part_b]
    else:
        rotation.records = None

    while True:
        if close_event.is_set():
            logger.info(f'receive signal to close data fetcher thread, closing now...')
            if cache is not None:
                cache.record.close()
            return

        # -------------------- Rotation Handling ----------------------------
        if rotation.medium == 'memory':
            # if rotate on memory, we simply move samples from data_queue to
            # rotation_queue until reaching the max size
            r_size = rotation_queue.qsize()
            if r_size <= rotation.max_size:
                for _ in range(rotation.max_size + 1 - r_size):
                    if not data_queue.empty():
                        minibatch = data_queue.get()
                        rotation_queue.put((1, minibatch))
                    else:
                        break
        else:
            # rotate on disk
            # records[0] is always for reading (if mode is read)
            # records[1] is always for writing (if mode is write)
            # in disk rotation, we use max_queue_size as the threshold for
            # rotation_queue
            # max_rotation_size is used as the number of samples in a cache
            # file
            r_size = rotation_queue.qsize()
            if r_size <= max_queue_size:
                if rotation.records[0].mode() == 'read':
                    # if there is rotation record for reading
                    if len(rotation.read_indices) > 0:
                        idx = rotation.read_indices.pop(0)
                        minibatch = rotation.records[0].read_index(idx)
                        # putting rot together with minibatch to
                        # prevent __next__() in asyncloader putting
                        # this minibatch batch back
                        # basically rotation is handled in this
                        # function
                        rotation_queue.put((rotation.max_rotation, minibatch))

                    # if we run out of indices, we need to increase
                    # rot_counter, which keeps track of how many rotations on
                    # this record has been made
                    if len(rotation.read_indices) == 0:
                        rotation.current_round += 1
                        if rotation.current_round >= rotation.max_rotation:
                            # if exceeding the number of rotations
                            # we dont read from this record anymore
                            # put this record into write mode
                            rotation.current_round = 0
                            rotation.read_indices = None
                            rotation.records[0].close()
                            rotation.records[0] = BinaryBlob(
                                binary_file=rotation.records[0].binary_file(),
                                index_file=rotation.records[0].index_file(),
                                mode='w',
                            )
                        else:
                            # if not exceeding the rotation limit
                            # recreate a list of indices again
                            rotation.read_indices = shuffle_indices(
                                start_idx=0,
                                stop_idx=len(rotation.records[0]),
                                nearby_shuffle=nearby_shuffle,
                            )

                if rotation.records[1].mode() == 'write':
                    # if 2nd record is in write mode, we need get from
                    # data_queue and write to records
                    if not data_queue.empty():
                        minibatch = data_queue.get()
                        rotation.records[1].write_index(rotation.write_idx, minibatch)
                        # if 1st record in write mode, we need to also put
                        # this minibatch into the rotation queue
                        if rotation.records[0].mode() == 'write':
                            # queued_indices keep track of which samples
                            # have been sent to the rotation queue
                            rotation.ignored_indices.append(rotation.write_idx)
                            rotation_queue.put((rotation.max_rotation, minibatch))

                        rotation.write_idx += 1

                    # check if we reach max_rotation_size
                    # then we need to close this record
                    if rotation.write_idx == rotation.max_size:
                        rotation.write_idx = 0
                        rotation.records[1].close()
                        # then open again in read mode
                        rotation.records[1] = BinaryBlob(
                            binary_file=rotation.records[1].binary_file(),
                            index_file=rotation.records[1].index_file(),
                            mode='r',
                        )

                if rotation.records[0].mode() == 'write' and rotation.records[1].mode() == 'read':
                    # swap position
                    rotation.records = rotation.records[::-1]
                    # then create a list of indices for future readout
                    rotation.read_indices = shuffle_indices(
                        start_idx=0,
                        stop_idx=len(rotation.records[0]),
                        nearby_shuffle=nearby_shuffle,
                    )
                    # remove those that are already in the rotation queue
                    rotation.read_indices = [
                        i for i in rotation.read_indices if i not in rotation.ignored_indices
                    ]
                    # reset queued_indices
                    rotation.ignored_indices = []


        # --------------------- Data Handling From Server --------------------
        # --------------------- This includes caching on client side ---------
        if cache is None:
            from_server = True
        else:
            if current_epoch == 0:
                # 1st epoch, if record in read mode
                # it means cache exists, just need to read from cache
                if cache.record.mode() == 'read':
                    from_server = False
                else:
                    from_server = True
            else:
                if current_epoch % cache.update_frequency == 0:
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
                    if minibatch_count == 0 and cache is not None and cache.record.mode() == 'read':
                        # close
                        cache.record.close()
                        # then open again for writing
                        cache.record = BinaryBlob(
                            cache.record.binary_file(),
                            cache.record.index_file(),
                            mode='w',
                        )

                    leftover, minibatch = clients[client_indices[0]].get()
                    if cache is not None:
                        cache.record.write_index(minibatch_count, minibatch)

                    if pin_memory:
                        minibatch = pin_data_memory(minibatch)

                    data_queue.put(minibatch)
                    minibatch_count += 1

                    if leftover == 0:
                        # remove the client from the list
                        client_indices.pop(0)
                        if len(client_indices) == 0:
                            # if empty, reinitialize
                            client_indices = list(range(nb_client))
                            # then reset the counter
                            minibatch_count = 0
                            current_epoch += 1
                            if cache is not None:
                                # if record is not None, we also need to close
                                # the record to finish the writing process
                                cache.record.close()
                                # then open record again in read mode
                                cache.record = BinaryBlob(
                                    cache.record.binary_file(),
                                    cache.record.index_file(),
                                    mode='r',
                                )
                    else:
                        # rotate the list
                        client_indices.append(client_indices.pop(0))
                else:
                    # if reading from cache file
                    minibatch = cache.record.read_index(minibatch_count)

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
        self.start_clients(packet_size, client_wait_time, nb_retry)
        self.is_client_available = True

        # counter to track minibatch
        self.minibatch_count = 0

        # start the thread to reconstruct data and put them into a queue
        self.data_queue = Queue()
        self.rotation_queue = Queue()
        self.device = device

        # use python threading
        self.fetcher_close_event = threading.Event()
        self.data_thread = threading.Thread(
            target=fetch_data,
            args=(
                self.clients,
                self.data_queue,
                self.rotation_queue,
                max_queue_size,
                self.fetcher_close_event,
                pin_memory,
                self.cache,
                self.rotation,
                nearby_shuffle,
            )
        )
        self.data_thread.start()
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

            if side == 'client' and rotation.medium == 'disk' and self.cache is not None:
                msg = (
                    'Caching on client side and rotation on disk can lead to unncessary overhead; ',
                    'Consider using memory as rotation medium and caching on client; ',
                    'Or using disk as rotation medium and no caching (or caching on server side);'
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
                self.fetcher_close_event.set()
                self.data_thread.join()

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

        while True:
            if self.rotation.max_rotation > 1:
                # rotation is enabled
                if (self.rotation_queue.qsize() >= self.rotation.min_size or self.minibatch_count > 0):
                    # has some minimum elements
                    readout = self.rotation_queue.get()
                    if readout is None:
                        error = self.rotation_queue.get()
                        self.close()
                        logger.warning(f'got error: {error}')
                        raise RuntimeError(error)

                    counter, minibatch = readout
                    if counter < self.rotation.max_rotation:
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
