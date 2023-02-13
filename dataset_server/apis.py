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

# size of the header
INTERCOM_HEADER_LEN = 4

def reconstruct_data(clients, data_queue, max_queue_size, close_event, device, pin_memory):
    # get the sample from the 1st client in the client list
    nb_client = len(clients)
    indices = list(range(nb_client))

    if device is None:
        # handle the case when no device is specified
        while True:
            if close_event.is_set():
                logger.info(f'receive signal to close reconstruction process')
                return

            if data_queue.qsize() < max_queue_size:
                try:
                    leftover, minibatch = clients[indices[0]].get()
                    if pin_memory:
                        data_queue.put([x.pin_memory() for x in minibatch])
                    else:
                        data_queue.put(minibatch)

                    if leftover == 0:
                        # remove the client from the list
                        indices.pop(0)
                        if len(indices) == 0:
                            # if empty, reinitialize
                            indices = list(range(nb_client))
                    else:
                        # rotate the list
                        indices.append(indices.pop(0))

                except RuntimeError as e:
                    # put None to data queue
                    # then put error message
                    data_queue.put(None)
                    data_queue.put(str(e))
                    for client in clients:
                        client.close()
                    raise RuntimeError(str(e))
            time.sleep(0.001)
    else:
        # create a buffer
        internal_queue = Queue()
        while True:
            if close_event.is_set():
                return

            if data_queue.qsize() < max_queue_size:
                try:
                    leftover, minibatch = clients[indices[0]].get()
                    if pin_memory:
                        data_queue.put([x.pin_memory().to(device, non_blocking=True) for x in minibatch])
                    else:
                        data_queue.put([x.to(device, non_blocking=True) for x in minibatch])

                    if leftover == 0:
                        # remove the client from the list
                        indices.pop(0)
                        if len(indices) == 0:
                            # if empty, reinitialize
                            indices = list(range(self.nb_client))
                    else:
                        # rotate the list
                        indices.append(indices.pop(0))

                except RuntimeError as e:
                    # put None to data queue
                    # then put error message
                    data_queue.put(None)
                    data_queue.put(str(e))
                    for client in clients:
                        client.close()
                    raise RuntimeError(str(e))
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
            while True:
                try:
                    self.socket.connect((self.hostname, self.port))
                    logger.info(f'connected to server {self.hostname} at port {self.port}')
                    break
                except Exception as e:
                    time.sleep(self.wait_time)
                    count += 1
                    if count >= self.nb_retry:
                        logger.warning(f'failed to connect after retrying {self.nb_retry} times')
                        logger.warning('terminating now!!!')

            # disable naggle algorithm
            #self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

            # now read the size of the dataset (number of minibatches)
            # the size is sent as 4 bytes
            self.size = int.from_bytes(self.socket.recv(4), 'big')
            self.write_socket('ok')

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
    ):
        try:
            dataset_tmp = dataset_class(**dataset_params)
            del dataset_tmp
        except Exception as error:
            logger.warning('failed to construct the dataset with the following error')
            raise error

        # start the servers
        logger.info('starting services, this will take a while')
        self.start_servers(
            dataset_class,
            dataset_params,
            batch_size,
            nb_servers,
            start_port,
            max_queue_size,
            shuffle,
            packet_size,
        )

        # start clients
        self.start_clients(packet_size)

        # counter to track minibatch
        self.minibatch_count = 0

        self.is_closed = False

        # start the thread to reconstruct data and put them into a queue
        self.data_queue = Queue()
        self.close_event = threading.Event()
        self.data_thread = threading.Thread(
            target=reconstruct_data,
            args=(self.clients, self.data_queue, max_queue_size, self.close_event, device, pin_memory)
        )
        self.data_thread.start()

        # wait a bit to generate some samples before returning
        logger.info(f'waiting {wait_time} seconds to preload some samples')
        time.sleep(wait_time)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def start_clients(self, packet_size):
        """
        start the socket clients
        """
        self.clients = []
        self.sizes = []
        self.total_minibatch = 0
        self.client_indices = []

        for index, port in enumerate(self.ports):
            new_client = DatasetClient(index, port, packet_size)
            self.clients.append(new_client)
            self.sizes.append(len(new_client))
            self.total_minibatch += len(new_client)
            self.client_indices.append(index)

        self.nb_client = len(self.sizes)

    def close(self):
        if not self.is_closed:
            logger.info('closing the AsyncDataLoader instance')
            self.close_event.set()
            self.data_thread.join()
            for client in self.clients:
                client.close()

            for server in self.servers:
                server.kill()

            self.is_closed = True

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
    ):
        """
        start the dataset servers
        """

        class_name = dataset_class.__name__
        dataset_module_file = inspect.getfile(dataset_class)

        # dump the file
        random_name = list(string.ascii_lowercase)
        random.shuffle(random_name)
        dataset_params_file = os.path.join(tempfile.gettempdir(), ''.join(random_name))
        with open(dataset_params_file, 'wb') as fid:
            dill.dump({'class_name': class_name, 'params': dataset_params}, fid, recurse=True)

        # start the server
        self.servers = []
        self.ports = []
        for server_idx in range(nb_servers):
            logger.info(f'starting dataset server {server_idx +1}')
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
                ]
            )
            time.sleep(5)
            self.servers.append(process)
            self.ports.append(start_port)
            start_port += 1


    def __len__(self):
        return self.total_minibatch

    def __iter__(self):
        return self

    def __next__(self):
        if self.minibatch_count >= self.total_minibatch:
            self.minibatch_count = 0
            raise StopIteration

        while True:
            if not self.data_queue.empty():
                minibatch = self.data_queue.get()
                if minibatch is not None:
                    break
                else:
                    error = self.data_queue.get()
                    self.close()
                    logger.warning(f'got error: {error}')
                    raise RuntimeError(error)
            else:
                time.sleep(0.001)

        # increase the minibatch counter
        self.minibatch_count += 1

        return minibatch

class AsyncDataset(AsyncDataLoader):
    def __init__(
        self,
        dataset_module_file: str,
        dataset_params: dict,
        nb_servers: int,
        start_port: int,
        max_queue_size: int,
        device=None,
        pin_memory=False,
    ):
        super().__init__(
            dataset_module_file=dataset_module_file,
            dataset_params=dataset_params,
            batch_size=1,
            nb_servers=nb_servers,
            start_port=start_port,
            max_queue_size=max_queue_size,
            device=device,
            pin_memory=pin_memory,
        )
