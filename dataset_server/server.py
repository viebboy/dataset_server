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
import queue
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
from dataset_server.common import BinaryBlob, shuffle_indices, Property


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
        # locks
        self.locks.index_changed = asyncio.Lock()
        self.locks.check_record = asyncio.Lock()

        # list of tasks
        self.tasks.tcp_server = None
        self.tasks.generate_sample_without_rotation = None
        self.tasks.generate_sample_with_rotation = None
        self.tasks.generate_minibatch = None
        self.tasks.start_batching = None
        self.tasks.rotate_data_on_disk = None
        self.tasks.rotate_data_on_memory = None
        self.tasks.cache_rotation = None
        self.tasks.load_dataset = None

        self.dataset_length = Queue()

        # this holds lists of samples to be concatenated
        self.sample_queue = Queue()

        # this holds ready minibatches to be sent
        # this queue is shared by both the telcom thread and the main thread
        self.minibatch_queue = Queue()

        # current_minibatch is a list that holds samples waiting to be batched
        self.current_minibatch = [[]]
        self.batcher_created = False

        self.record = None
        self.cache_update_frequency = 0
        self.cache_binary_file = None
        self.cache_index_file = None


    @verbose
    async def enter__(self):
        self.tasks.tcp_server = await reCreate(self.tasks.tcp_server, self.tcp_server__)
        self.tasks.load_dataset = await reCreate(self.tasks.load_dataset, self.load_dataset__)

    @verbose
    async def start_batching(self):
        try:
            # move the current minibatch into sample_queue for batching
            # start generate minibatch task
            # reset the current minibatch
            self.sample_queue.put(self.current_minibatch.pop(0))
            # launch the task to perform batching and send it to children
            if not self.batcher_created:
                self.tasks.generate_minibatch = await reCreate(
                    self.tasks.generate_minibatch,
                    self.generate_minibatch__
                )

                self.batcher_created = True
            else:
                # schedule the task to run
                self.tasks.generate_minibatch = await reSchedule(
                    self.generate_minibatch__
                )

            # reset
            self.current_minibatch = [[]]
        except asyncio.CancelledError:
            self.print_warning('start_batching', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'start_batching',
                str(e)
            )


    def require_reading_from_dataset(self):
        if self.rotation.medium in [None, 'memory']:
            epoch_idx = self.rotation.write_epoch_idx
        else:
            epoch_idx = self.rotation.epoch_idx

        if self.cache is None:
            from_dataset = True
        else:
            if epoch_idx == 0:
                if self.cache.record.mode() == 'read':
                    from_dataset = False
                else:
                    from_dataset = True
            else:
                if (epoch_idx % self.cache.update_freq) == 0:
                    from_dataset = True
                else:
                    from_dataset = False

        return from_dataset

    @verbose
    async def cache_rotation__(self):
        """
        this function writes data to self.rotation.records
        and start batching

        self.locks.check_record
        self.locks.check_indices_to_ignore

        self.rotation.disk.sample_counter = 0
        self.rotation.disk.sample_write_idx = 0
        self.rotation.disk.current_round = 0
        self.rotation.disk.indices_to_read = []
        self.rotation.disk.indices_to_ignore = []

        IMPORTANT: we might want to continue writing as long as there's cache space
        but we might not want to add samples to current_minibatch and batching because
        the queue might be full so we NEED to check the size of minibatch_queue like
        rotate_data_on_disk

        """
        try:
            async with self.locks.check_record:
                if self.rotation.records[1].mode() == 'write':
                    can_write = True
                    if self.rotation.records[0].mode() == 'write':
                        # if 1st record is also in write mode and
                        # minibatch_queue is not full, we need to put data in
                        # current_minibatch[0] and perform batching
                        need_batching = not self.is_minibatch_queue_full()
                    else:
                        need_batching = False
                else:
                    can_write = False
                    need_batching = False

            if can_write:
                qsize = self.rotation.queue.qsize()
                for _ in range(qsize):
                    try:
                        minibatch = self.rotation.queue.get(block=False)
                        if minibatch is None:
                            # finalize the writing stage
                            await self.finalize_rotation_file_writing()

                            # in addition, batching if needed
                            if need_batching and len(self.current_minibatch[0]) > 0:
                                await self.start_batching()

                        else:
                            # otherwise, write this sample to record
                            self.rotation.records[1].write_index(
                                self.rotation.disk.sample_write_idx,
                                minibatch
                            )

                            # batching if needed
                            if need_batching:
                                self.current_minibatch[0].append(minibatch)
                                # add the current index to the list of indices to ignore
                                # here we use a list and a queue to avoid using
                                # a lock all the time.
                                self.rotation.disk.indices_to_ignore_list[0].append(
                                    self.rotation.disk.sample_write_idx
                                )
                                # then start batching if reaching the batch size
                                if len(self.current_minibatch[0]) == self.batch_size:
                                    await self.start_batching()

                            # increase counter
                            self.rotation.disk.sample_write_idx += 1

                            # check if reaching the maximum size of a record
                            if self.rotation.disk.sample_write_idx == self.rotation.max_size:
                                # finalize the writing stage
                                await self.finalize_rotation_file_writing()

                    except queue.Empty():
                        # empty read
                        await asyncio.sleep(0.001)
            else:
                # sleep a bit before rescheduling
                await asyncio.sleep(0.01)

            # reschedule
            self.tasks.cache_rotation = await reSchedule(self.cache_rotation__)

        except asyncio.CancelledError:
            self.print_warning('cache_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'cache_rotation__',
                str(e)
            )

    @verbose
    async def finalize_rotation_file_writing(self):
        try:
            # check if we need to flush indices to ignore
            if len(self.rotation.disk.indices_to_ignore_list[0]) > 0:
                self.rotation.disk.indices_to_ignore.put(
                    self.rotation.disk.indices_to_ignore_list.pop(0)
                )
                # make a new list for the next writing round
                self.rotation.disk.indices_to_ignore_list.append([])

            # reset the counter
            self.rotation.disk.sample_write_idx = 0
            # close the record to finalize writing
            # and open again for reading
            self.rotation.records[1].close()
            self.rotation.records[1] = BinaryBlob(
                binary_file=self.rotation.records[1].binary_file(),
                index_file=self.rotation.records[1].index_file(),
                mode='r',
            )

            # check if record[0] in write mode --> swap
            async with self.locks.check_record:
                if self.rotation.records[0].mode() == 'write' and self.rotation.records[1].mode() == 'read':
                    self.rotation.records = self.rotation.records[::-1]

        except asyncio.CancelledError:
            self.print_warning('finalize_rotation_file_writing', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'finalize_rotation_file_writing',
                str(e)
            )

    def is_minibatch_queue_full(self):
        # check if whether the numbef of minibatches to be sent is too much
        if self.rotation.mb_counter > 0:
            is_full = False
            self.rotation.mb_counter -= 1
        else:
            self.rotation.mb_counter = self.rotation.mb_max_size
            if self.minibatch_queue.qsize() > self.max_queue_size:
                is_full = True
            else:
                is_full = False

        return is_full

    @verbose
    async def rotate_data_on_disk__(self):
        """
        this function reads data from self.rotation.records
        and start batching
        and keep track of how many times a sample has been rotated

        self.locks.check_record

        self.rotation.disk.sample_counter = 0
        self.rotation.disk.current_round = 0
        self.rotation.disk.indices_to_read = []
        self.rotation.disk.indices_to_ignore = Queue

        """
        try:
            async with self.locks.check_record:
                if self.rotation.records[0].mode() == 'read':
                    can_read = True
                else:
                    can_read = False
                is_full = self.is_minibatch_queue_full()

            if not can_read:
                await asyncio.sleep(0.01)

            if is_full:
                await asyncio.sleep(0.001)

            if not is_full and can_read:
                # now check if there is indices to read
                if len(self.rotation.disk.indices_to_read) == 0:
                    # generate them
                    indices = shuffle_indices(
                        0,
                        len(self.rotation.records[0]),
                        self.nearby_shuffle
                    )
                    # and also check if there is any indices to ignore
                    if not self.rotation.disk.indices_to_ignore.empty():
                        try:
                            indices_to_ignore = self.rotation.disk.indices_to_ignore.get(block=False)
                            indices = [i for i in indices if i not in indices_to_ignore]
                        except queue.Empty:
                            pass
                    self.rotation.disk.indices_to_read = indices

                # compute the maximum number of minibatches that we can
                # generate based on max_queue_size
                max_minibatch = max(1, self.max_queue_size - self.minibatch_queue.qsize())
                minibatch_count = 0
                # loop through the leftover indices and read from record
                for _ in range(len(self.rotation.disk.indices_to_read)):
                    idx = self.rotation.disk.indices_to_read.pop(0)
                    minibatch = self.rotation.records[0].read_index(idx)

                    # increase the counter
                    # this counter is used to track if we reach the end of epoch
                    self.rotation.disk.sample_counter += 1
                    self.current_minibatch[0].append(minibatch)

                    # if there is enough sample to batch, then start batching
                    if len(self.current_minibatch[0]) == self.batch_size:
                        await self.start_batching()
                        minibatch_count += 1

                        if minibatch_count >= max_minibatch:
                            break

                # now check if indices_to_read is empty
                # then increase the round counter
                if len(self.rotation.disk.indices_to_read) == 0:
                    self.rotation.disk.current_round += 1
                    self.rotation.disk.indices_to_read = []

                    if self.rotation.disk.current_round >= self.rotation.max_rotation:
                        # reset the counter
                        self.rotation.disk.current_round = 0

                        # use the lock to handle the record
                        async with self.locks.check_record:
                            # close the record
                            self.rotation.records[0].close()
                            # and open again in write mode
                            self.rotation.records[0] = BinaryBlob(
                                binary_file = self.rotation.records[0].binary_file(),
                                index_file = self.rotation.records[0].index_file(),
                                mode='w',
                            )

                        # also need to check if this record is the last one in
                        # the epoch
                        if self.rotation.disk.sample_counter == self.rotation.disk.total_sample:
                            # reset
                            self.rotation.disk.sample_counter = 0
                            # then start batching
                            # here we need to check if there is any sample to
                            # batch because if the total number of samples is
                            # divisible by batch size then after completing the
                            # record, we also dont have any samples left in
                            # current_minibatch
                            if len(self.current_minibatch[0]) > 0:
                                await self.start_batching()

            self.tasks.rotate_data_on_disk = await reSchedule(
                self.rotate_data_on_disk__
            )

        except asyncio.CancelledError:
            self.print_warning('rotate_data_on_disk__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'rotate_data_on_disk__',
                str(e)
            )


    @verbose
    async def rotate_data_on_memory__(self):
        """
        this function is used when rotation on memory is specified
        or no rotation is specified
        """
        """
            self.rotation.queue = [Queue(), Queue()]
            self.rotation.queue_max_size = max(1, 0.5 * (self.max_queue_size)) * self.batch_size
            self.rotation.queue_counter = max(1, 0.5 * (self.max_queue_size)) * self.batch_size
            self.rotation.read_queue_idx = 0
            self.rotation.read_sample_idx = 0
            self.rotation.read_epoch_idx = 0
            self.rotation.write_queue_idx = 0
            self.rotation.write_sample_idx = 0
            self.rotation.write_epoch_idx = 0
            self.rotation.total_write = self.total_sample
            self.rotation.total_read = self.total_sample * self.rotation.count
            self.rotation.write_allowed = True
        """
        try:
            # reset counter if needed
            if self.rotation.read_sample_idx == self.rotation.total_read:
                # reaching the total sample, change the index now
                self.rotation.read_sample_idx = 0
                # change to index of the queue to read from
                async with self.locks.index_changed:
                    self.rotation.read_queue_idx = 1 - self.rotation.read_queue_idx
                    # if there are samples in current_minibatch, we need to
                    # batch them too. Note that we dont want to wait until this
                    # reaches the batch size because these samples belong to
                    # the current epoch and we are about to move to a new epoch
                    if len(self.current_minibatch[0]) > 0:
                        # batch these leftover samples
                        await self.start_batching()

            # check if whether the numbef of minibatches to be sent is too much
            if self.rotation.mb_counter > 0:
                is_full = False
                self.rotation.mb_counter -= 1
            else:
                self.rotation.mb_counter = self.rotation.mb_max_size
                if self.minibatch_queue.qsize() > self.max_queue_size:
                    is_full = True
                else:
                    is_full = False

            if not is_full and self.rotation.queue[self.rotation.read_queue_idx].qsize() >= self.rotation.min_size:
                # note that we dont enforce less than max size here because we
                # still need to move samples from the queue to
                # current_minibatch and start batching
                # max_size is only enforced in generate_sample
                # if there is data enough to read
                try:
                    # try to get sample
                    counter, sample = self.rotation.queue[self.rotation.read_queue_idx].get(block=False)
                    self.rotation.read_sample_idx += 1
                    self.current_minibatch[0].append(sample)

                    if len(self.current_minibatch[0]) == self.batch_size:
                        await self.start_batching()

                    if counter < self.rotation.max_rotation:
                        # if not reaching the maximum rotation
                        # put back into the queue
                        self.rotation.queue[self.read_queue_idx].put((counter + 1, sample), block=False)

                except queue.Empty:
                    await asyncio.sleep(0.001)
                    pass
            else:
                await asyncio.sleep(0.001)

            self.tasks.rotate_data_on_memory = await reSchedule(
                self.rotate_data_on_memory__
            )

        except asyncio.CancelledError:
            self.print_warning('rotate_data_on_memory__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'rotate_data_on_memory__',
                str(e)
            )


    @verbose
    async def generate_sample_without_rotation__(self):
        """
        this function is used when rotation on disk is specified
        """
        try:
            # check if whether the queue is too full
            if self.rotation.queue_counter > 0:
                is_full = False
                self.rotation.queue_counter -= 1
            else:
                self.rotation.queue_counter = self.rotation.queue_max_size
                if self.rotation.queue.qsize() > self.rotation.queue_max_size:
                    is_full = True
                else:
                    is_full = False

            if not is_full:
                # check whether sample is read from dataset or from cache
                # get a sample and put in the current list
                if self.require_reading_from_dataset():
                    # if 1st sample, then close the current record that has
                    # been used for reading and open again for writing
                    reset = (
                        self.rotation.sample_idx == 0 and
                        self.cache is not None and
                        self.cache.record.mode() == 'read'
                    )
                    if reset:
                        self.cache.record.close()
                        self.cache.record = BinaryBlob(
                            self.cache.bin_file,
                            self.cache.idx_file,
                            mode='w'
                        )

                    sample_idx = self.indices[self.rotation.sample_idx]
                    sample = self.dataset[sample_idx]

                    # if not None, then write to record
                    if self.cache is not None:
                        self.cache.record.write_index(sample_idx, sample)

                    # put the sample into the write queue (queue for writing)
                    # 1 represents the number of time that a sample has been
                    # rotated
                    self.rotation.queue.put(sample)

                    # increment the counter that marks how many samples have
                    # been written to the queue
                    self.rotation.sample_idx += 1

                    # if reaching the total samples
                    if self.rotation.sample_idx == self.rotation.total_sample:
                        # reset the sample index counter and epoch counter
                        self.rotation.sample_idx = 0
                        self.ratotion.epoch_idx += 1

                        # we need to put None into the queue to signify the end
                        # of epoch
                        self.rotation.queue.put(None)

                        # close the record that was in writing mode
                        # and open again in reading mode
                        if self.cache is not None:
                            self.cache.record.close()
                            self.cache.record = BinaryBlob(
                                self.cache.bin_file,
                                self.cache.idx_file,
                                mode='r',
                            )

                        # shuffle the indices if needed
                        if self.shuffle:
                            start_idx = min(self.indices)
                            stop_idx = max(self.indices) + 1
                            self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)
                else:
                    # if we could read from record instead of from dataset
                    sample_idx = self.indices[self.rotation.sample_idx]
                    sample = self.record.read_index(sample_idx)
                    self.rotation.queue.put(sample)

                    self.rotation.sample_idx += 1
                    if self.rotation.sample_idx == self.rotation.total_sample:
                        # reset the counter
                        self.rotation.sample_idx = 0
                        self.rotation.epoch_idx += 1

            else:
                # data queue is full, sleep a bit
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
        """
                self.rotation.queue = [Queue(), Queue()]
                self.rotation.queue_max_size = max(1, 0.5 * (self.max_queue_size)) * self.batch_size
                self.rotation.queue_counter = max(1, 0.5 * (self.max_queue_size)) * self.batch_size
                self.rotation.read_queue_idx = 0
                self.rotation.read_sample_idx = 0
                self.rotation.read_epoch_idx = 0
                self.rotation.write_queue_idx = 0
                self.rotation.write_sample_idx = 0
                self.rotation.write_epoch_idx = 0
                self.rotation.total_write = self.total_sample
                self.rotation.total_read = self.total_sample * self.rotation.count
                self.rotation.write_allowed = True
        """

        try:
            # check if whether the queue is too full
            if self.rotation.queue_counter > 0:
                is_full = False
                self.rotation.queue_counter -= 1
            else:
                self.rotation.queue_counter = self.rotation.queue_max_size
                if self.rotation.queue[self.rotation.write_queue_idx].qsize() > self.rotation.queue_max_size:
                    is_full = True
                else:
                    is_full = False

            if not is_full and self.rotation.write_allowed:
                # check whether sample is read from dataset or from cache
                # get a sample and put in the current list
                if self.require_reading_from_dataset():
                    # if 1st sample, then close the current record that has
                    # been used for reading and open again for writing
                    reset = (
                        self.rotation.write_sample_idx == 0 and
                        self.cache is not None and
                        self.cache.record.mode() == 'read'
                    )
                    if reset:
                        self.cache.record.close()
                        self.cache.record = BinaryBlob(
                            self.cache.bin_file,
                            self.cache.idx_file,
                            mode='w'
                        )

                    sample_idx = self.indices[self.rotation.write_sample_idx]
                    sample = self.dataset[sample_idx]

                    # if not None, then write to record
                    if self.cache is not None:
                        self.cache.record.write_index(sample_idx, sample)

                    # put the sample into the write queue (queue for writing)
                    # 1 represents the number of time that a sample has been
                    # rotated
                    self.rotation.queue[self.rotation.write_queue_idx].put((1, sample))

                    # increment the counter that marks how many samples have
                    # been written to the queue
                    self.rotation.write_sample_idx += 1

                    # if reaching the total samples
                    if self.rotation.write_sample_idx == self.rotation.total_write:
                        # reset the sample index counter and epoch counter
                        self.rotation.write_sample_idx = 0
                        self.ratotion.write_epoch_idx += 1

                        async with self.locks.index_changed:
                            # change the index of queue to write
                            self.rotation.write_queue_idx = int(1 - self.rotation.write_queue_idx)
                            # if read and write index are the same but read and write epoch are different
                            # then it means we need to stop writing temporarily
                            self.rotation.write_allowed = not (
                                self.rotation.write_queue_idx == self.rotation.read_queue_idx and
                                self.rotation.read_epoch_idx != self.rotation.write_epoch_idx
                            )

                        # close the record that was in writing mode
                        # and open again in reading mode
                        if self.cache is not None:
                            self.cache.record.close()
                            self.cache.record = BinaryBlob(
                                self.cache.bin_file,
                                self.cache.idx_file,
                                mode='r',
                            )

                        # shuffle the indices if needed
                        if self.shuffle:
                            start_idx = min(self.indices)
                            stop_idx = max(self.indices) + 1
                            self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)
                else:
                    # if we could read from record instead of from dataset
                    sample_idx = self.indices[self.rotation.write_sample_idx]
                    sample = self.record.read_index(sample_idx)
                    self.rotation.queue[self.rotation.write_queue_idx].put((1, sample))

                    self.rotation.write_sample_idx += 1
                    if self.rotation.write_sample_idx == self.rotation.total_write:
                        # reset the counter
                        self.rotation.write_sample_idx = 0
                        self.rotation.write_epoch_idx += 1

                        async with self.locks.index_changed:
                            # change the index of queue to write
                            self.rotation.write_queue_idx = int(1 - self.rotation.write_queue_idx)
                            # if read and write index are the same but read and write epoch are different
                            # then it means we need to stop writing temporarily
                            self.rotation.write_allowed = not (
                                self.rotation.write_queue_idx == self.rotation.read_queue_idx and
                                self.rotation.read_epoch_idx != self.rotation.write_epoch_idx
                            )
            elif is_full:
                # data queue is full, sleep a bit
                await asyncio.sleep(0.001)

            elif not self.rotation.write_allowed:
                # check again if writing is now allowed
                async with self.locks.index_changed:
                    self.rotation.write_allowed = (
                        self.rotation.write_queue_idx != self.rotation.read_queue_idx or
                        self.rotation.read_epoch_idx == self.rotation.write_epoch_idx
                    )

            self.tasks.generate_sample_with_rotation = await reSchedule(
                self.generate_sample_with_rotation__
            )

        except asyncio.CancelledError:
            self.print_warning('generate_sample_with_rotation__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'generate_sample_with_rotation__',
                str(e)
            )


    @verbose
    async def warn_and_exit(self, function_name, warning):
        if self.record is not None:
            self.record.close()
        await super().warn_and_exit(function_name, warning)


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
                rotation_setting = settings['rotation_setting']

            logger.debug(f'complete loading parameters for dataset')

            # create dataset
            # the module must have a Dataset class
            # and its parameters are passed as keyword arguments
            constructor = getattr(module, class_name)
            self.dataset = constructor(**params)

            # handle caching
            if cache_setting is not None and cache_setting['cache_side'] == 'server':
                self.cache = Property()
                self.cache.bin_file = cache_setting['cache_prefix'] + '{:09d}.bin'.format(self.server_index)
                self.cache.idx_file = cache_setting['cache_prefix'] + '{:09d}.idx'.format(self.server_index)
                # if files exist
                if os.path.exists(self.cache.bin_file) and os.path.exists(self.cache.idx_file):
                    # if rewrite, we delete the files
                    if cache_setting['rewrite']:
                        os.remove(self.cache.bin_file)
                        os.remove(self.cache.idx_file)
                        self.cache.record = BinaryBlob(self.cache.bin_file, self.cache.idx_file, mode='w')
                    else:
                        self.cache.record = BinaryBlob(self.cache.bin_file, self.cache.idx_file, mode='r')
                else:
                    self.cache.record = BinaryBlob(self.cache.bin_file, self.cache.idx_file, mode='w')
                # if update frequency = K --> the cache files are rewritten
                # every K epochs
                self.cache.update_freq = cache_setting['update_frequency']
                if self.cache.update_freq < 2:
                    await self.warn_and_exit(
                        'load_dataset__',
                        f'cache update frequency must be at least 2, received: {self.cache.update_freq}'
                    )
            else:
                self.cache = None

            # handle rotation
            self.rotation = Property()
            if rotation_setting is None:
                self.rotation.max_rotation = 1
                self.rotation.min_size = 1
                self.rotation.max_size = max(1, self.max_queue_size) * self.batch_size
                self.rotation.medium = None
            else:
                self.rotation.medium = rotation_setting['rotation_medium']
                self.rotation.max_rotation = rotation_setting['rotation']
                self.rotation.min_size = rotation_setting['min_rotation_size']
                self.rotation.max_size = rotation_setting['max_rotation_size']
                if self.rotation.medium == 'disk':
                    self.rotation.prefix = rotation_setting['rotation_file_prefix']
                    part_a = BinaryBlob(
                        self.rotation.prefix + f'_{self.server_idx}_A.bin',
                        self.rotation.prefix + f'_{self.server_idx}_A.idx',
                        mode='w',
                    )
                    part_b = BinaryBlob(
                        self.rotation.prefix + f'_{self.server_idx}_B.bin',
                        self.rotation.prefix + f'_{self.server_idx}_B.idx',
                        mode='w',
                    )
                    self.rotation.records = [part_a, part_b]
                    # create counters for samples on disk
                    self.rotation.disk = Property()
                    self.rotation.disk.sample_write_idx = 0
                    self.rotation.disk.sample_counter = 0
                    self.rotation.disk.current_round = 0
                    self.rotation.disk.indices_to_read = []
                    self.rotation.disk.indices_to_ignore_list = [[]]
                    self.rotation.disk.indices_to_ignore = Queue()


            sample_per_server = math.ceil(len(self.dataset) / self.total_server)
            start_idx = self.server_index * sample_per_server
            stop_idx = min(len(self.dataset), (self.server_index + 1) * sample_per_server)
            self.total_sample = stop_idx - start_idx
            self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)

            nb_mini_batch = int(np.ceil(self.total_sample / self.batch_size))
            self.dataset_length.put(nb_mini_batch)

            while not self.server_started:
                await asyncio.sleep(0.1)

            with open(self.status_file, 'w') as fid:
                fid.write('ready')

            logger.info('server completes loading dataset')

            # start the task to prefetch samples
            if self.rotation.medium in [None, 'memory']:
                # perform initialization for this case
                # we need 2 queues to correctly rotate data between epochs
                # without mixing them
                self.rotation.queue = [Queue(), Queue()]
                # this is to keep track of the queue size
                self.rotation.queue_max_size = self.rotation.max_size
                self.rotation.queue_counter = self.rotation.max_size
                # this is to keep track of the number of minibatches
                self.rotation.mb_max_size = max(1, self.max_queue_size) * self.batch_size
                self.rotation.mb_counter = max(1, self.max_queue_size) * self.batch_size
                # there are 2 queues, read and write queue
                # these are counters for read queue
                self.rotation.read_queue_idx = 0
                self.rotation.read_sample_idx = 0
                self.rotation.read_epoch_idx = 0
                # these are counters for write queue
                self.rotation.write_queue_idx = 0
                self.rotation.write_sample_idx = 0
                self.rotation.write_epoch_idx = 0
                # total number of samples that need to be written to
                # write_queue in an epoch
                self.rotation.total_write = self.total_sample
                # total number of samples that need to be read from read_queue in an epoch
                self.rotation.total_read = self.total_sample * self.rotation.max_rotation
                # whether writing to write_queue is allowed
                self.rotation.write_allowed = True

                self.tasks.generate_sample_with_rotation = await reCreate(
                    self.tasks.generate_sample_with_rotation,
                    self.generate_sample_with_rotation__
                )

                await asyncio.sleep(0.01)
                self.tasks.rotate_data_on_memory = await reCreate(
                    self.tasks.rotate_data_on_memory,
                    self.rotate_data_on_memory__
                )

            else:
                # for rotation on disk, we dont need to take care of rotation
                # when generating sample
                self.rotation.queue = Queue()
                # when rotation on disk, the queue max size is equal to the
                # number of samples at max_queue_size (that is, multiplied by
                # batch size because max_queue_size has a unit of minibatch
                # rotation.max_size is the total number of samples that are
                # written to a binary blob

                # this is to keep track of the queue size
                self.rotation.queue_max_size = max(1, self.max_queue_size) * self.batch_size
                self.rotation.queue_counter = max(1, self.max_queue_size) * self.batch_size

                # this is to keep track of the number of minibatches
                self.rotation.mb_max_size = max(1, self.max_queue_size) * self.batch_size
                self.rotation.mb_counter = max(1, self.max_queue_size) * self.batch_size

                self.rotation.sample_idx = 0
                self.rotation.epoch_idx = 0
                self.rotation.total_sample = self.total_sample
                self.rotation.disk.total_sample = self.total_sample * self.rotation.max_rotation

                # start task to read samples from dataset
                self.tasks.generate_sample_without_rotation = await reCreate(
                    self.tasks.generate_sample_without_rotation,
                    self.generate_sample_without_rotation__
                )

                await asyncio.sleep(0.01)
                # start task to write samples to cache file and start batching if needed
                self.tasks.cache_rotation = await reCreate(
                    self.tasks.cache_rotation,
                    self.cache_rotation__
                )

                await asyncio.sleep(0.01)
                # start task to read samples from cache file and start batching if needed
                self.tasks.rotate_data_on_disk = await reCreate(
                    self.tasks.rotate_data_on_disk,
                    self.rotate_data_on_disk__
                )

        except asyncio.CancelledError:
            self.print_warning('load_dataset__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit('load_dataset__', str(e))

    @verbose
    async def exit__(self):
        self.tasks.tcp_server = await delete(self.tasks.tcp_server)
        self.tasks.load_dataset = await delete(self.tasks.load_dataset)

        self.tasks.generate_sample_with_rotation = await delete(
            self.tasks.generate_sample_with_rotation
        )
        self.tasks.generate_minibatch_without_rotation = await delete(
            self.tasks.generate_minibatch_without_rotation
        )
        self.tasks.generate_minibatch = await delete(
            self.tasks.generate_minibatch
        )
        self.tasks.rotate_data_on_disk = await delete(
            self.tasks.rotate_data_on_disk
        )
        self.tasks.rotate_data_on_memory = await delete(
            self.tasks.rotate_data_on_memory
        )
        self.tasks.cache_rotation = await delete(
            self.tasks.cache_rotation
        )
        self.tasks.start_batching = await delete(
            self.tasks.start_batching
        )

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
