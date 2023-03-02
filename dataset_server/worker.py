"""
worker.py: data processing implementation using async
-----------------------------------------------------


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
from loguru import logger

from dataset_server.common import (
    BinaryBlob,
    shuffle_indices,
    Property,
    TaskThread,
    INTERCOM_HEADER_LEN,
    object_to_bytes,
    bytes_to_object,
)


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


class DatasetLoader(TaskThread):
    """
    Dataloader class
    """
    def __init__(
        self,
        shared_memory,
        read_pipe,
        write_pipe,
        dataset_module_file,
        dataset_params_file,
        nb_loader,
        loader_index,
        batch_size,
        max_queue_size,
        shuffle,
        nearby_shuffle,
        name,
    ):
        self.shared_memory = shared_memory
        self.max_shared_mem_size = len(shared_memory.buf)
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.dataset_module_file = dataset_module_file
        self.dataset_params_file = dataset_params_file
        self.loader_index = loader_index
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.shuffle = shuffle
        self.nearby_shuffle = nearby_shuffle

        super(DatasetLoader, self).__init__(parent = None, name=name)


    def initVars__(self):
        # locks
        self.locks.queue_index_changed = asyncio.Lock()
        self.locks.check_record = asyncio.Lock()
        self.locks.change_send_status = asyncio.Lock()

        # list of tasks
        self.tasks.generate_sample_without_rotation = None
        self.tasks.generate_sample_with_rotation = None
        self.tasks.generate_minibatch = None
        self.tasks.check_parent_message = None
        self.tasks.start_batching = None
        self.tasks.rotate_data_on_disk = None
        self.tasks.rotate_data_on_memory = None
        self.tasks.cache_rotation = None
        self.tasks.load_dataset = None

        # this holds lists of samples to be concatenated
        self.sample_queue = Queue()

        # this holds ready minibatches to be sent
        # this queue is shared by both the telcom thread and the main thread
        self.minibatch_queue = Queue()

        # current_minibatch is a list that holds samples waiting to be batched
        self.current_minibatch = [[]]
        self.batcher_created = False

        # flags to keep track of shared memory
        # initially false because we need to check response from parent
        self.can_send = False


    @verbose
    async def enter__(self):
        self.tasks.load_dataset = await reCreate(self.tasks.load_dataset, self.load_dataset__)

    @verbose
    async def start_batching(self):
        """
        this function takes out the samples in the current minibatch
        and put them in self.sample_queue, which is monitored by another repetitive task
        called generate_minibatch__
        """
        try:
            # move the current minibatch into sample_queue for batching
            # reset the current minibatch
            self.sample_queue.put(self.current_minibatch.pop(0))

            # reset
            self.current_minibatch = [[]]

        except asyncio.CancelledError:
            self.print_warning('start_batching', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'start_batching',
                str(e)
            )


    @verbose
    async def generate_minibatch__(self):
        # batching the samples and send to children
        try:
            if not self.sample_queue.empty() and self.can_send:
                samples = self.sample_queue.get()
                minibatch = concatenate_list(samples)
                minibatch = dill.dumps(minibatch)
                if len(minibatch) > self.max_shared_mem_size:
                    await self.warn_and_exit(
                        'generate_minibatch__',
                        'got a minibatch that is bigger than shared memory space'
                    )
                else:
                    # put bytes into shared_memory
                    self.shared_memory.buf[:len(minibatch)] = minibatch
                    # then send the number of bytes to read
                    self.write_pipe.send(
                        {
                            'title': 'minibatch_length',
                            'content': len(minibatch),
                        }
                    )

                    # change the send status
                    async with self.locks.change_send_status:
                        self.can_send = False
            else:
                await asyncio.sleep(0.001)

            self.tasks.generate_minibatch = await reSchedule(self.generate_minibatch__)

        except asyncio.CancelledError:
            self.print_warning('generate_minibatch__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'generate_minibatch__',
                str(e)
            )


    @verbose
    async def check_parent_message__(self):
        """
        check the message from parent process through pipe
        """
        try:
            if self.read_pipe.poll():
                response = self.read_pipe.recv()
                if response == 'can_send':
                    async with self.locks.change_send_status:
                        self.can_send = True
                elif response == 'close':
                    # receive close signal
                    logger.info(f'receive close signal from parent process; closing now...')
                    await self.clean_and_exit()
            else:
                await asyncio.sleep(0.001)

            self.tasks.check_parent_message = await reSchedule(self.check_parent_message__)

        except asyncio.CancelledError:
            self.print_warning('check_parent_message__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'check_parent_message__',
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
        this task is used to write rotation data to disk
        """
        try:
            async with self.locks.check_record:
                if self.rotation.records[1].mode() == 'write':
                    can_write = True
                    if self.rotation.records[0].mode() == 'write':
                        # if 1st record is also in write mode and
                        # minibatch_queue is not full, we need to put data in
                        # current_minibatch[0] and perform batching
                        need_batching = not self.is_sample_queue_full()
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

    def is_sample_queue_full(self):
        if self.sample_queue.qsize() > self.max_queue_size:
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
                is_full = self.is_sample_queue_full()

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
                max_minibatch = max(1, self.max_queue_size - self.sample_queue.qsize())
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
        this task performs data rotation on memory (applied when no rotation or rotation on memory)
        basically this task will read samples from a read queue in self.rotation.queue
        if a sample has been not been rotated enough, it will be put back to self.rotation.queue
        """
        try:
            # reset counter if needed
            # read_sample_idx is a counter to keep track of how many samples
            # have been processed in the current epoch
            if self.rotation.read_sample_idx == self.rotation.total_read:
                # reaching the total sample, change the index now
                self.rotation.read_sample_idx = 0
                # change to index of the queue to read from
                async with self.locks.queue_index_changed:
                    #logger.warning(f'changing read queue index from {self.rotation.read_queue_idx} to ?')
                    self.rotation.read_queue_idx = 1 - self.rotation.read_queue_idx
                    # if there are samples in current_minibatch, we need to
                    # batch them too. Note that we dont want to wait until this
                    # reaches the batch size because these samples belong to
                    # the current epoch and we are about to move to a new epoch
                    if len(self.current_minibatch[0]) > 0:
                        # batch these leftover samples
                        await self.start_batching()

            is_full = self.is_sample_queue_full()

            # check if the read_queue has minimum size to start reading
            # this is because when we read a sample, we will put it back to the
            # queue --> we want to read when queue has at least some samples to
            # avoid rotating a sample K continuous times
            # note: we dont need to use the lock because
            # generate_sample_with_rotation only modifies write_queue_idx, not
            # the read_queue_idx
            can_read = (
                (self.rotation.queue[self.rotation.read_queue_idx].qsize() >= self.rotation.min_size) or
                self.rotation.read_sample_idx > 0
            )

            if not is_full and can_read:
                # note that we dont enforce less than max size here because we
                # still need to move samples from the queue to
                # current_minibatch and start batching
                # max_size is only enforced in generate_sample
                # if there is data enough to read
                try:
                    # try to get sample without blocking
                    counter, sample = self.rotation.queue[self.rotation.read_queue_idx].get(block=False)
                    self.rotation.read_sample_idx += 1
                    self.current_minibatch[0].append(sample)

                    if len(self.current_minibatch[0]) == self.batch_size:
                        await self.start_batching()

                    if counter < self.rotation.max_rotation:
                        # if not reaching the maximum rotation
                        # put back into the queue
                        self.rotation.queue[self.rotation.read_queue_idx].put((counter + 1, sample), block=False)

                except queue.Empty:
                    # if fails to read from the queue, wait a bit then try
                    # again later
                    await asyncio.sleep(0.001)
            else:
                # if cannot read, we wait a bit before rescheduling
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
                        self.rotation.epoch_idx += 1

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
        read sample from dataset and put them into the write queue of self.rotation.queue
        there are 2 queues: one for reading and one for writing
        in this function, data loaded from dataset will be written to the write queue
        we need 2 queues because the data will also be rotated in these queues and
        we need the rotation in an epoch to complete before the rotation in the next epoch starts

        this task is scheduled to repeat indefinitely until canceled
        this task is used when there is no rotation (max_rotation=1) or rotation is done on memory
        """

        try:
            # check if whether the queue for writing is too full
            # here we use queue_counter to avoid checking the size of a queue every time this task is run
            # we start the queue_counter with the maximum size of the queue,
            # then decreases every time this task is run
            # when queue_counter == 0 --> we start the actual checking
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
                        # reset the counter and change the index of queue to write
                        await self.change_write_queue_index()

                        # close the record that was in writing mode
                        # and open again in reading mode
                        if self.cache is not None:
                            self.cache.record.close()
                            self.cache.record = BinaryBlob(
                                self.cache.record.binary_file(),
                                self.cache.record.index_file(),
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
                    sample = self.cache.record.read_index(sample_idx)
                    self.rotation.queue[self.rotation.write_queue_idx].put((1, sample))

                    self.rotation.write_sample_idx += 1

                    if self.rotation.write_sample_idx == self.rotation.total_write:
                        # reset the counter and change the index of queue to write
                        await self.change_write_queue_index()

            elif is_full:
                # data queue is full, sleep a bit
                #logger.warning(f'rotation.queue: write queue is full')
                await asyncio.sleep(0.001)

            elif not self.rotation.write_allowed:
                #logger.warning('writing to queue is not allowed')
                # check again if writing is now allowed
                await asyncio.sleep(0.001)
                async with self.locks.queue_index_changed:
                    self.rotation.write_allowed = (
                        self.rotation.write_queue_idx != self.rotation.read_queue_idx or
                        self.rotation.read_epoch_idx == self.rotation.write_epoch_idx
                    )

            # reschedule this task
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
    async def change_write_queue_index(self):
        try:
            # reset the counter
            self.rotation.write_sample_idx = 0
            self.rotation.write_epoch_idx += 1

            async with self.locks.queue_index_changed:
                # change the index of queue to write
                #logger.warning(f'changing write queue index from: {self.rotation.write_queue_idx}')
                self.rotation.write_queue_idx = 1 - self.rotation.write_queue_idx
                # if read and write index are the same but read and write epoch are different
                # then it means we need to stop writing temporarily
                # this part that accesses read_queue_idx is the one that
                # requires the lock
                self.rotation.write_allowed = not (
                    self.rotation.write_queue_idx == self.rotation.read_queue_idx and
                    self.rotation.read_epoch_idx != self.rotation.write_epoch_idx
                )

        except asyncio.CancelledError:
            self.print_warning('change_write_queue_index', 'got canceled')
        except Exception as e:
            await self.warn_and_exit(
                'change_write_queue_index',
                str(e)
            )

    @verbose
    async def warn_and_exit(self, function_name, warning):
        # clean up cacche
        if self.cache is not None:
            self.cache.record.close()

        # clean up rotation
        if self.rotation.medium == 'disk':
            self.rotation.records[0].close()
            self.rotation.records[1].close()

        # close shared memory
        self.shared_memory.close()

        # close the write pipe
        self.write_pipe.send({'title': 'close_with_error', 'content': warning})
        self.write_pipe.close()

        await super().warn_and_exit(function_name, warning)

    @verbose
    async def clean_and_exit(self, function_name, warning):
        # clean up cacche
        if self.cache is not None:
            self.cache.record.close()

        # clean up rotation
        if self.rotation.medium == 'disk':
            self.rotation.records[0].close()
            self.rotation.records[1].close()

        # close shared memory
        self.shared_memory.close()

        # close the write pipe
        self.write_pipe.close()

        # then stop everything
        await self.stop()


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
                    f'the dataset loader might fail to load dataset because dependent modules are not loaded',
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
            if cache_setting is not None:
                self.cache = Property()
                self.cache.bin_file = cache_setting['prefix'] + '{:09d}.bin'.format(self.loader_index)
                self.cache.idx_file = cache_setting['prefix'] + '{:09d}.idx'.format(self.loader_index)
                self.cache.record = BinaryBlob(self.cache.bin_file, self.cache.idx_file, mode='w')

                # if update frequency = K --> the cache files are rewritten
                # every K epochs
                self.cache.update_freq = cache_setting['update_frequency']
            else:
                self.cache = None

            # handle rotation
            self.rotation = Property()
            self.rotation.medium = rotation_setting['medium']
            self.rotation.max_rotation = rotation_setting['rotation']
            self.rotation.min_size = self.batch_size + 1
            if self.rotation_medium == 'memory':
                self.rotation.max_size = self.batch_size * self.max_queue_size
            else:
                self.rotation.max_size = rotation_setting['size']

            if self.rotation.medium == 'disk':
                self.rotation.prefix = rotation_setting['prefix']
                part_a = BinaryBlob(
                    self.rotation.prefix + f'_{self.loader_index}_A.bin',
                    self.rotation.prefix + f'_{self.loader_index}_A.idx',
                    mode='w',
                )
                part_b = BinaryBlob(
                    self.rotation.prefix + f'_{self.loader_index}_B.bin',
                    self.rotation.prefix + f'_{self.loader_index}_B.idx',
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

            sample_per_loader = math.ceil(len(self.dataset) / self.nb_loader)
            start_idx = self.loader_index * sample_per_loader
            stop_idx = min(len(self.dataset), (self.loader_index + 1) * sample_per_loader)
            self.total_sample = stop_idx - start_idx

            # create a list of indices for reading from dataset
            if self.shuffle:
                self.indices = shuffle_indices(start_idx, stop_idx, self.nearby_shuffle)
            else:
                self.indices = list(range(start_idx, stop_idx))

            self.total_minibatch = int(np.ceil(self.total_sample * self.rotation.max_rotation / self.batch_size))

            self.write_pipe.send({'title': 'readiness', 'content': True})
            logger.info('completes loading dataset')

            # start the task to prefetch samples
            if self.rotation.medium == 'memory':
                # perform initialization for this case
                # we need 2 queues to correctly rotate data between epochs
                # without mixing them
                self.rotation.queue = [Queue(), Queue()]
                # this is to keep track of the queue size
                self.rotation.queue_max_size = max(1, int(0.5 * self.max_queue_size)) * self.batch_size
                self.rotation.queue_counter = max(1, int(0.5 * self.max_queue_size)) * self.batch_size
                # this is to keep track of the number of minibatches
                self.rotation.mb_max_size = self.max_queue_size * self.batch_size
                self.rotation.mb_counter = self.max_queue_size * self.batch_size
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
                self.rotation.queue_max_size = max(1, int(0.5 * self.max_queue_size)) * self.batch_size
                self.rotation.queue_counter = max(1, int(0.5 * self.max_queue_size)) * self.batch_size

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

            # start to generate minibatches
            self.tasks.generate_minibatch = await reCreate(
                self.tasks.generate_minibatch,
                self.generate_minibatch__
            )

            self.tasks.check_parent_message = await reCreate(
                self.tasks.check_parent_message,
                self.check_parent_message__
            )

        except asyncio.CancelledError:
            self.print_warning('load_dataset__', 'got canceled')
        except Exception as e:
            await self.warn_and_exit('load_dataset__', str(e))

    @verbose
    async def exit__(self):
        self.tasks.load_dataset = await delete(self.tasks.load_dataset)

        self.tasks.generate_sample_with_rotation = await delete(
            self.tasks.generate_sample_with_rotation
        )
        self.tasks.generate_sample_without_rotation = await delete(
            self.tasks.generate_sample_without_rotation
        )
        self.tasks.generate_minibatch = await delete(
            self.tasks.generate_minibatch
        )
        self.tasks.check_parent_message = await delete(
            self.tasks.check_parent_message
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
