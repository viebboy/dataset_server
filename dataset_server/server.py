"""
server.py: server implementation for dataloader via socket
----------------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-07-29
* Version: 0.0.1

License
-------
Apache License 2.0


"""

import torch
import asyncio, logging, json, socket, dill, random
import traceback
import time
import sys
from queue import Queue
import math
import numpy as np
import os
from task_thread import (
    reCreate,
    reSchedule,
    delete,
    verbose,
    signals
)
import copy
from task_thread import TaskThread as BaseTaskThread
from loguru import logger
from dataset_server.common import BinaryBlob, shuffle_indices


# size of the header
INTERCOM_HEADER_LEN = 4


def object_to_bytes(dic):
    """Turn dic into bytes & append the bytes with the bytes length
    """
    bytes_ = dill.dumps(dic)
    le = len(bytes_)
    bytes_ = le.to_bytes(INTERCOM_HEADER_LEN, "big") + bytes_
    return bytes_

def bytes_to_object(byte_arr):
    obj = dill.loads(byte_arr)
    return obj

def concatenate_list(inputs, device=None):
    """
    This function is used to concatenate
    a list of nested lists of numpy array (or torch) to a nested list of numpy arrays or torch

    For example,
    inputs = [x_1, x_2, ..., x_N]
    each x_i is a nested list of numpy arrays
    for example,
    x_i = [np.ndarray, [np.ndarray, np.ndarray]]

    the outputs should be
    outputs = [np.ndarray, [np.ndarray, np.ndarray]]
    in which each np.ndarray is a concatenation of the corresponding element
    from the same position
    """

    def _create_nested_list(x, data):
        if isinstance(x, (list, tuple)):
            # there is a nested structure
            # create sub list for each item
            for _ in range(len(x)):
                data.append([])
            for idx, item in enumerate(x):
                _create_nested_list(item, data[idx])
        else:
            # do nothing to the input list
            next

    def _process_sample(x, data):
        if isinstance(x, (list, tuple)):
            for idx, item in enumerate(x):
                _process_sample(item, data[idx])
        else:
            data.append(x)

    # create an empty nested list based on the structure of the 1st sample
    outputs = []
    _create_nested_list(inputs[0], outputs)

    # then process each sample
    for sample in inputs:
        _process_sample(sample, outputs)

    def _concatenate_item(x):
        if isinstance(x, list) and isinstance(x[0], torch.Tensor):
            if torch.numel(x[0]) == 1:
                result = torch.cat(x)
            else:
                result = torch.cat([item.unsqueeze(0) for item in x], dim=0)

            if device is not None:
                return result.to(device)
            else:
                return result
        elif isinstance(x, list) and isinstance(x[0], np.ndarray):
            return np.concatenate([np.expand_dims(item, 0) for item in x], axis=0)
        elif isinstance(x, list) and isinstance(x[0], (float, int)):
            return np.asarray(x).flatten()
        elif isinstance(x, list) and isinstance(x[0], list):
            return [_concatenate_item(item) for item in x]
        else:
            raise RuntimeError('Failed to concatenate a list of samples generated from dataset')

    return _concatenate_item(outputs)



class TaskThread(BaseTaskThread):
    """
    A modified base TaskThread class that implements some functionalities
    """

    def __init__(self, name, parent=None):
        super().__init__(parent=parent)
        self.name = name
        self.parent = parent

    async def warn_and_exit(self, function_name, message=''):
        logger.warning(
            f'{self.getInfo()} {function_name}: face issue {message}'
        )
        logger.warning(
            f'{self.getInfo()} {function_name}: exit now...'
        )
        await self.stop()

    def getId(self):
        return str(id(self))

    def getInfo(self):
        return f"<{self.name} {self.getId()}>"

    def print_info(self, function_name, message):
        logger.info(f'{self.getInfo()} {function_name}: {message}')

    def print_warning(self, function_name, message):
        logger.warning(f'{self.getInfo()} {function_name}: {message}')


class ServerConnectionHandler(TaskThread):
    """
    This TaskThread is used to handle whenever a new client is connected to the server
    """
    def __init__(
        self,
        minibatch_queue,
        dataset_length,
        parent = None,
        reader = None,
        writer = None,
        packet_size = 125000,
        name='ServerConnectionHandler'
    ):
        # max buffer is the number of samples that could stay in the stream
        # buffer
        self.minibatch_queue = minibatch_queue
        self.dataset_length = dataset_length
        self.reader = reader
        self.writer = writer
        self.packet_size = packet_size
        super(ServerConnectionHandler, self).__init__(parent = parent, name=name)

    def initVars__(self):
        """
        register the basic tasks here
        """
        self.tasks.read_socket = None
        self.tasks.write_socket = None
        self.tasks.send_minibatch = None
        self.tasks.monitor_response = None
        self.tasks.send_dataset_info = None
        self.obj_queue = Queue()
        self.reset_read_package_state__(True)

        # variable to keep track of the stream flushing
        self.is_flushed = False
        self.sender_task_created = False

    @verbose
    async def send_minibatch__(self):
        try:
            if not self.minibatch_queue.empty() and self.is_flushed:
                minibatch = self.minibatch_queue.get()
                await self.write_socket__(minibatch)
                self.is_flushed = False

            # reschedule the task
            await asyncio.sleep(0.001)
            self.tasks.send_minibatch = await reSchedule(self.send_minibatch__)

        except asyncio.CancelledError:
            self.print_warning('send_minibatch__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'send_minibatch__',
                str(e)
            )

    @verbose
    async def send_dataset_info__(self):
        try:
            if not self.dataset_length.empty():
                length = self.dataset_length.get()
                byte_rep = length.to_bytes(INTERCOM_HEADER_LEN, 'big')
                self.writer.write(byte_rep)

                # launch the monitor response task
                self.tasks.monitor_response = await reCreate(
                    self.tasks.monitor_response,
                    self.monitor_response__
                )
            else:
                # dataset has not been loaded
                # reschedule this task until dataset length is available
                await asyncio.sleep(1)
                self.tasks.send_dataset_info = await reSchedule(
                    self.send_dataset_info__
                )
        except asyncio.CancelledError:
            self.print_warning('send_dataset_info__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit('send_dataset_info__', str(e))

    @verbose
    async def monitor_response__(self):
        try:
            if not self.obj_queue.empty():
                response = self.obj_queue.get()
                # handle response after sending a mini-batch
                if response == 'ok':
                    self.is_flushed = True

                    if not self.sender_task_created:
                        self.sender_task_created = True
                        self.tasks.send_minibatch = await reCreate(
                            self.tasks.send_minibatch,
                            self.send_minibatch__
                        )
                else:
                    await self.warn_and_exit(
                        'monitor_response__',
                        f'Unknown response: {response}'
                    )

            # sleep a bit then reschedule
            await asyncio.sleep(0.001)
            self.tasks.monitor_response = await reSchedule(self.monitor_response__)

        except asyncio.CancelledError:
            self.print_warning('monitor_response__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'monitor_response__',
                str(e)
            )

    @verbose
    async def enter__(self):
        """
        read_socket() and write_socket() are not created in initialization
        """
        logger.info(f"enter__ : {self.getInfo()}")

        # launch task to read from socket
        self.tasks.read_socket = await reCreate(
            self.tasks.read_socket,
            self.read_socket__
        )

        self.tasks.send_dataset_info = await reCreate(
            self.tasks.send_dataset_info,
            self.send_dataset_info__
        )


    @verbose
    async def exit__(self):
        self.tasks.read_socket = await delete(self.tasks.read_socket)
        self.tasks.write_socket = await delete(self.tasks.write_socket)
        self.tasks.send_minibatch = await delete(self.tasks.send_minibatch)
        self.tasks.monitor_response = await delete(self.tasks.monitor_response)
        self.tasks.send_dataset_info = await delete(self.tasks.send_dataset_info)
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception as e:
            logger.warning(f'Failed to close StreamWriter with this error: {e}')
        logger.debug("exit__: bye!")

    async def read_socket__(self):
        try:
            try:
                packet = await self.reader.read(self.packet_size)
                """a word of warning here: try to tread as much bytes as you can with the
                asyncio StreamReader instance (here self.reader) per re-scheduling,
                in order to keep the rescheduling frequency reasonable
                """
            except Exception as e:
                await self.warn_and_exit('read_socket__', str(e))
            else:
                if len(packet) > 0:
                    # all good!  keep on reading = reschedule this
                    logger.debug(f"read_socket__: got packet with size {len(packet)}")
                    # send the payload to parent:
                    await self.handle_read_packet__(packet)
                    """you can re-schedule with a moderate frequency, say, 100 times per second,
                    but try to keep the re-scheduling frequency "sane"
                    """
                    self.tasks.read_socket = await reSchedule(self.read_socket__); return
                else:
                    await self.warn_and_exit('read_socket__', 'client has stopped the connection')

        except asyncio.CancelledError:
            self.print_warning('read_socket__', 'got canceled')

        except Exception as e:
            await self.warn_and_exit('read_socket__', str(e))

    def reset_read_package_state__(self, clearbuf = False):
        self.left = INTERCOM_HEADER_LEN
        self.len = 0
        self.header = True
        if clearbuf:
            self.read_buf = bytes(0)

    async def handle_read_packet__(self, packet):
        """packet reconstructions into blobs of certain length
        """
        if packet is not None:
            self.read_buf += packet
        if self.header:
            if len(self.read_buf) >= INTERCOM_HEADER_LEN:
                self.len = int.from_bytes(self.read_buf[0:INTERCOM_HEADER_LEN], "big")
                self.header = False # we got the header info (length)
                if len(self.read_buf) > INTERCOM_HEADER_LEN:
                    # sort out the remaining stuff
                    await self.handle_read_packet__(None)
        else:
            if len(self.read_buf) >= (INTERCOM_HEADER_LEN + self.len):
                # correct amount of bytes have been obtained
                payload = bytes_to_object(
                    self.read_buf[INTERCOM_HEADER_LEN:INTERCOM_HEADER_LEN + self.len]
                )
                # put reconstructed objects into a queue
                self.obj_queue.put(payload)
                # prepare state for next blob
                if len(self.read_buf) > (INTERCOM_HEADER_LEN + self.len):
                    # there's some leftover here for the next blob..
                    self.read_buf = self.read_buf[INTERCOM_HEADER_LEN + self.len:]
                    self.reset_read_package_state__()
                    # .. so let's handle that leftover
                    await self.handle_read_packet__(None)
                else:
                    # blob ends exactly
                    self.reset_read_package_state__(clearbuf=True)

    async def write_socket__(self, obj=None):
        """
        Send an object to the client
        """
        try:
            # sequentially write a block of self.packet_size bytes
            if obj is not None:
                # convert to byte array
                self.write_buf = object_to_bytes(obj)
                self.write_start = 0
                # then reschedule to call this task again
                self.tasks.write_socket = await reSchedule(self.write_socket__)
                return
            else:
                # this part is the actual writing part
                # we will write one blob of self.packet_size at a time
                # and allow returning the control to the main loop if needed
                if self.write_start < len(self.write_buf):
                    write_stop = min(len(self.write_buf), self.write_start + self.packet_size)
                    try:
                        self.writer.write(self.write_buf[self.write_start:write_stop])
                        await self.writer.drain()
                    except Exception as e:
                        logger.warning(f"write_socket__ : sending to client failed : {e}")
                        logger.info(f"write_socket__: tcp connection {self.getInfo()} terminated")
                        await self.stop()
                    else:
                        self.write_start = write_stop
                        self.tasks.write_socket = await reSchedule(self.write_socket__); return
                else:
                    return
        except asyncio.CancelledError:
            self.print_warning('write_socket__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit('write_socket__', str(e))


class DataloaderServer(TaskThread):
    """
    DataServer class
    """
    def __init__(
        self,
        status_file,
        dataset_module_file,
        dataset_params_file,
        total_server,
        server_index,
        batch_size,
        max_queue_size,
        shuffle,
        nearby_shuffle,
        name="DataloaderServer",
        retry_interval=5,
        port=5002,
        max_clients=1,
        device=None,
        packet_size=125000,
    ):
        if os.path.exists(status_file) and os.path.isfile(status_file):
            os.remove(status_file)

        self.status_file = status_file
        self.dataset_module_file = dataset_module_file
        self.dataset_params_file = dataset_params_file
        self.total_server = total_server
        self.server_index = server_index
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.shuffle = shuffle
        self.nearby_shuffle = nearby_shuffle
        self.retry_interval = retry_interval
        self.port = port
        self.max_clients = max_clients
        self.device = None if device is None else torch.device(device)
        self.packet_size = packet_size

        super(DataloaderServer, self).__init__(parent = None, name=name)


    def initVars__(self):
        self.server = None
        self.server_started = False

        # list of tasks
        self.tasks.tcp_server = None
        self.tasks.generate_sample_without_rotation = None
        self.tasks.generate_minibatch_without_rotation = None
        self.tasks.generate_sample_with_rotation = None
        self.tasks.generate_minibatch_with_rotation = None
        self.tasks.move_minibatch = None
        self.tasks.load_dataset = None

        self.dataset_length = Queue()

        # this holds lists of samples to be concatenated
        self.sample_queue = Queue()
        self.sample_queue_counter = self.max_queue_size

        # this holds ready minibatches to be sent
        # this queue is shared by both the telcom thread
        # and the main thread
        self.minibatch_queue = Queue()
        self.current_minibatch = [[]]
        self.batcher_created = False
        self.record = None
        self.cache_update_frequency = 0
        self.current_epoch = 0
        self.cache_binary_file = None
        self.cache_index_file = None


    @verbose
    async def enter__(self):
        self.tasks.tcp_server = await reCreate(self.tasks.tcp_server, self.tcp_server__)
        self.tasks.load_dataset = await reCreate(self.tasks.load_dataset, self.load_dataset__)


    @verbose
    async def move_minibatch(self):
        try:
            # move the current minibatch into sample_queue for batching
            # start generate minibatch task
            # reset the current minibatch

            self.sample_queue.put(self.current_minibatch.pop(0))
            # launch the task to perform batching and send it to children
            if not self.batcher_created:
                # if not yet started for the 1st time, then created
                if self.rotation_setting is None:
                    self.tasks.generate_minibatch_without_rotation = await reCreate(
                        self.tasks.generate_minibatch_without_rotation,
                        self.generate_minibatch_without_rotation__
                    )
                else:
                    self.tasks.generate_minibatch_with_rotation = await reCreate(
                        self.tasks.generate_minibatch_with_rotation,
                        self.generate_minibatch_with_rotation__
                    )

                self.batcher_created = True
            else:
                # schedule the task to run
                if self.rotation_setting is None:
                    self.tasks.generate_minibatch_without_rotation = await reSchedule(
                        self.generate_minibatch_without_rotation__
                    )
                else:
                    self.tasks.generate_minibatch_with_rotation = await reSchedule(
                        self.generate_minibatch_with_rotation__
                    )

            # reset
            self.current_minibatch = [[]]
        except asyncio.CancelledError:
            self.print_warning('move_minibatch_without_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'move_minibatch_without_rotation__',
                str(e)
            )


    def require_reading_from_dataset(self):
        if self.record is None:
            from_dataset = True
        else:
            if self.current_epoch == 0:
                # 1st epoch, if record in read mode
                # it means cache exists, just need to read from cache
                if self.record.mode() == 'read':
                    from_dataset = False
                else:
                    from_dataset = True
            else:
                if self.current_epoch % self.cache_update_frequency == 0:
                    from_dataset = True
                else:
                    from_dataset = False
        return from_dataset

    @verbose
    async def generate_sample_without_rotation__(self):
        try:
            if len(self.current_minibatch[0]) == self.batch_size:
                # reaching the size, need to start batching
                await self.move_minibatch_without_rotation()
            else:
                # check if the current queue size exceeds the max
                if self.sample_queue_counter > 0:
                    is_full = False
                    self.sample_queue_counter -= 1
                else:
                    self.sample_queue_counter = self.max_queue_size
                    if self.sample_queue.qsize() >= self.max_queue_size:
                        is_full = True
                    else:
                        is_full = False

                if not is_full:
                    # check whether sample is read from dataset or from cache
                    # get a sample and put in the current list
                    if self.require_reading_from_dataset():
                        # if 1st sample, then close the current record that has
                        # been used for reading and open again for writing
                        if self.cur_idx == 0 and self.record is not None and self.record.mode() == 'read':
                            self.record.close()
                            self.record = BinaryBlob(self.cache_binary_file, self.cache_index_file, mode='w')

                        idx = self.indices[self.cur_idx]
                        sample = self.dataset[idx]

                        # if not None, then write to record
                        if self.record is not None:
                            self.record.write_index(idx, sample)

                        self.current_minibatch[0].append(sample)

                        self.cur_idx += 1
                        if self.cur_idx == self.total_sample:
                            # this is the end of dataset
                            # we need to batch this left-over part
                            await self.move_minibatch()

                            # reset the sample index counter and epoch counter
                            self.cur_idx = 0
                            self.current_epoch += 1
                            # close the record that was in writing mode
                            # and open again in reading mode
                            if self.record is not None:
                                self.record.close()
                                self.record = BinaryBlob(
                                    self.cache_binary_file,
                                    self.cache_index_file,
                                    mode='r',
                                )

                            # shuffle the indices if needed
                            if self.shuffle:
                                start_idx = min(self.indices)
                                stop_idx = max(self.indices) + 1
                                self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)
                    else:
                        # if we could read from record
                        idx = self.indices[self.cur_idx]
                        sample = self.record.read_index(idx)
                        self.current_minibatch[0].append(sample)
                        self.cur_idx += 1
                        if self.cur_idx == self.total_sample:
                            # again, reaching the end of the dataset
                            # need to start batching
                            await self.move_minibatch()
                            # then reset the counter
                            self.cur_idx = 0
                            self.current_epoch += 1
                else:
                    await asyncio.sleep(0.001)

            self.tasks.generate_sample_without_rotation = await reSchedule(
                self.generate_sample_without_rotation__
            )

        except asyncio.CancelledError:
            self.print_warning('generate_sample_without_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'generate_sample_without_rotation__',
                str(e)
            )


    @verbose
    async def generate_sample_with_rotation__(self):
        try:
            if len(self.current_minibatch[0]) == self.batch_size:
                # reaching the size, need to start batching
                await self.move_minibatch()
            else:
                # check if the current queue size exceeds the max
                if self.minibatch_queue.qsize() <= self.max_queue_size:
                    # check whether sample is read from dataset or from cache
                    # get a sample and put in the current list
                    if self.require_reading_from_dataset():
                        # if 1st sample, then close the current record that has
                        # been used for reading and open again for writing
                        if self.cur_idx == 0 and self.record is not None and self.record.mode() == 'read':
                            self.record.close()
                            self.record = BinaryBlob(self.cache_binary_file, self.cache_index_file, mode='w')

                        idx = self.indices[self.cur_idx]
                        sample = self.dataset[idx]

                        # if not None, then write to record
                        if self.record is not None:
                            self.record.write_index(idx, sample)

                        self.current_minibatch[0].append(sample)

                        self.cur_idx += 1
                        if self.cur_idx == self.total_sample:
                            # this is the end of dataset
                            # we need to batch this left-over part
                            await self.move_minibatch_without_rotation()

                            # reset the sample index counter and epoch counter
                            self.cur_idx = 0
                            self.current_epoch += 1
                            # close the record that was in writing mode
                            # and open again in reading mode
                            if self.record is not None:
                                self.record.close()
                                self.record = BinaryBlob(
                                    self.cache_binary_file,
                                    self.cache_index_file,
                                    mode='r',
                                )

                            # shuffle the indices if needed
                            if self.shuffle:
                                start_idx = min(self.indices)
                                stop_idx = max(self.indices) + 1
                                self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)
                    else:
                        # if we could read from record
                        idx = self.indices[self.cur_idx]
                        sample = self.record.read_index(idx)
                        self.current_minibatch[0].append(sample)
                        self.cur_idx += 1
                        if self.cur_idx == self.total_sample:
                            # again, reaching the end of the dataset
                            # need to start batching
                            await self.move_minibatch_without_rotation()
                            # then reset the counter
                            self.cur_idx = 0
                            self.current_epoch += 1
                else:
                    await asyncio.sleep(0.001)

            self.tasks.generate_sample_without_rotation = await reSchedule(
                self.generate_sample_without_rotation__
            )

        except asyncio.CancelledError:
            self.print_warning('generate_sample_without_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'generate_sample_without_rotation__',
                str(e)
            )


    @verbose
    async def warn_and_exit(self, function_name, warning):
        if self.record is not None:
            self.record.close()
        await super().warn_and_exit(function_name, warning)


    @verbose
    async def generate_minibatch_without_rotation__(self):
        # batching the samples and send to children
        try:
            if not self.sample_queue.empty():
                samples = self.sample_queue.get()
                minibatch = concatenate_list(samples, self.device)
                self.minibatch_queue.put(minibatch)
        except asyncio.CancelledError:
            self.print_warning('generate_minibatch_without_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'generate_minibatch_without_rotation__',
                str(e)
            )

    @verbose
    async def load_dataset__(self):
        # load parameters first
        try:

            # then import the data set file
            module_name = os.path.basename(self.dataset_module_file).replace('.py', '')
            logger.debug(f'dataset module file: {self.dataset_module_file}, module name: {module_name}')

            import importlib.util
            spec = importlib.util.spec_from_file_location('custom_dataset', self.dataset_module_file)
            module = importlib.util.module_from_spec(spec)

            # check if there is name clashing
            if module_name not in sys.modules.keys():
                sys.modules[module_name] = module
            else:
                msg = (
                    f'name clashing: the dataset module file: {self.dataset_module_file} ',
                    f'has module name = {module_name}, which clashes with one of sys.modules. ',
                    f'the dataset server might fail to load dataset because dependent modules are not loaded',
                )
                logger.warning(''.join(msg))
                logger.warning(f'sys.modules contain the following modules: {sys.modules.keys()}')

            spec.loader.exec_module(module)
            logger.debug(f'complete executing module')

            logger.debug(f'loading dataset parameter file: {self.dataset_params_file}')
            with open(self.dataset_params_file, 'rb') as fid:
                settings = dill.load(fid)
                logger.debug('complete dill deserialization of dataset params')
                class_name = settings['class_name']
                params = settings['params']
                cache_setting = settings['cache_setting']

            logger.debug(f'complete loading parameters for dataset')

            # create dataset
            # the module must have a Dataset class
            # and its parameters are passed as keyword arguments
            constructor = getattr(module, class_name)
            self.dataset = constructor(**params)

            # handle caching
            if cache_setting is not None and cache_setting['cache_side'] == 'server':
                cache_binary_file = cache_setting['cache_prefix'] + '{:09d}.bin'.format(self.server_index)
                cache_index_file = cache_setting['cache_prefix'] + '{:09d}.idx'.format(self.server_index)
                # if files exist
                if os.path.exists(cache_index_file) and os.path.exists(cache_binary_file):
                    # if rewrite, we delete the files
                    if cache_setting['rewrite']:
                        os.remove(cache_index_file)
                        os.remove(cache_binary_file)
                        self.record = BinaryBlob(cache_binary_file, cache_index_file, mode='w')
                    else:
                        self.record = BinaryBlob(cache_binary_file, cache_index_file, mode='r')
                else:
                    self.record = BinaryBlob(cache_binary_file, cache_index_file, mode='w')
                # if update frequency = K --> the cache files are rewritten
                # every K epochs
                self.cache_update_frequency = cache_setting['update_frequency']
                if self.cache_update_frequency < 2:
                    await self.warn_and_exit(
                        'load_dataset__',
                        f'cache update frequency must be at least 2, received: {self.cache_update_frequency}'
                    )
                self.cache_binary_file = cache_binary_file
                self.cache_index_file = cache_index_file


            sample_per_server = math.ceil(len(self.dataset) / self.total_server)
            start_idx = self.server_index * sample_per_server
            stop_idx = min(len(self.dataset), (self.server_index + 1) * sample_per_server)

            self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)

            self.cur_idx = 0
            self.total_sample = stop_idx - start_idx
            nb_mini_batch = int(np.ceil(self.total_sample / self.batch_size))
            self.dataset_length.put(nb_mini_batch)

            while not self.server_started:
                await asyncio.sleep(0.1)

            with open(self.status_file, 'w') as fid:
                fid.write('ready')

            logger.info('server completes loading dataset')

            # start the task to prefetch samples
            self.tasks.generate_sample = await reCreate(self.tasks.generate_sample, self.generate_sample__)

        except asyncio.CancelledError:
            self.print_warning('load_dataset__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit('load_dataset__', str(e))

    @verbose
    async def exit__(self):
        self.tasks.tcp_server = await delete(self.tasks.tcp_server)
        # tasks with rotation
        self.tasks.generate_sample_with_rotation = await delete(
            self.tasks.generate_sample_with_rotation
        )
        self.tasks.generate_minibatch_with_rotation = await delete(
            self.tasks.generate_minibatch_with_rotation
        )
        # tasks without rotation
        self.tasks.generate_sample_without_rotation = await delete(
            self.tasks.generate_sample_without_rotation
        )
        self.tasks.generate_minibatch_without_rotation = await delete(
            self.tasks.generate_minibatch_without_rotation
        )
        self.tasks.move_minibatch = await delete(self.tasks.move_minibatch)
        self.tasks.load_dataset = await delete(self.tasks.load_dataset)

    @verbose
    async def tcp_server__(self):
        """
        This handle a new connection by creating a ServerConnectionHandler task
        """
        try:
            logger.info(f'{self.getInfo()}: server starting at port {self.port}')
            self.server = await asyncio.start_server(self.new_connection_handler__, "", self.port)

        except asyncio.CancelledError:
            self.print_warning('tcp_server__', 'got canceled')
            self.server = None
            self.server_started = False

        except Exception as e:
            logger.warning(f"tcp_server__: failed with {e}")
            logger.warning(f"tcp_server__: will try again in {self.retry_interval} secs")
            await asyncio.sleep(self.retry_interval)
            self.server_started = False
            self.tasks.tcp_server = await reSchedule(self.tcp_server__)
        else:
            logger.info("tcp_server__ : new server waiting")
            self.server_started = True

    @verbose
    async def new_connection_handler__(self, reader, writer):
        """
        handle a new connection
        """
        logger.info(f"new_connection_handler__ : new connection for {self.getInfo()}")

        if len(self.children) > self.max_clients:
            logger.warning(
                f"new_connection_handler__ : max number of connections is {self.max_clients}"
            )
            return

        child_connection = ServerConnectionHandler(
            minibatch_queue=self.minibatch_queue,
            dataset_length=self.dataset_length,
            reader=reader,
            writer=writer,
            parent=self,
            packet_size=self.packet_size,
            name=self.name,
        )
        await child_connection.run()
        await self.addChild(child_connection)
