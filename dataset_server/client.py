"""
client.py: client implementation to reconstruct data from server
----------------------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

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
from loguru import logger
import asyncio
from task_thread import (
    reCreate,
    reSchedule,
    delete,
    verbose,
    signals
)

from dataset_server.common import (
    BinaryBlob,
    TaskThread,
    INTERCOM_HEADER_LEN,
    bytes_to_object,
    object_to_bytes,
)


class DatasetClient(TaskThread):
    """
    Client side of dataset server
    This handles minibatch reconstruction from TCP connection
    This also handle caching on client side
    """
    def __init__(self, data_queue, event_queue, **kwargs):
        # multiprocessing queue to exchange data with another process
        self.data_queue = data_queue
        self.event_queue = event_queue

        # connection related params
        self.retry_interval = kwargs['retry_interval']
        self.port = kwargs['port']
        self.max_retry = kwargs['nb_retry']
        self.packet_size = kwargs['packet_size']
        self.host_adr = 'localhost'

        self.cache = kwargs['cache']
        self.max_queue_size = kwargs['max_queue_size']
        self.nearby_shuffle = kwargs['nearby_shuffle']
        self.client_index = kwargs['client_index']

        super(DatasetClient, self).__init__(
            parent=None,
            name='DatasetClient',
        )

    def initVars__(self):
        self.tasks.read_socket = None
        self.tasks.write_socket = None
        self.tasks.init_connection = None
        # payload includes: nb of minibatches, actual minibatches
        # process payload doesnt need to write back to server because
        # read_socket handles this
        self.tasks.process_payload = None
        # this process is used to monitor the event queue
        self.tasks.monitor_event = None

        # reader and writer objects
        self.reader, self.writer = None, None

        # flags to keep track of connection status
        self.is_connected = False
        self.cur_attempt = 0

        # calling reset is necessary to initialize reader
        self.reset_read_package_state__(True)

        # this queue is for holding received objects
        self.received_payload = Queue()

        # total minibatch is the 1st payload sent by server
        self.total_minibatch = None
        self.minibatch_idx = 0

        # counter to check whether payload queue is full
        self.read_socket_counter = self.max_queue_size

        self.fps_count= 0
        self.start_time = None

        # initialize counter for cache
        if self.cache is not None:
            self.cache.current_epoch = 0
            binary_file = self.cache.prefix + f'_{self.client_index}.bin'
            index_file = self.cache.prefix + f'_{self.client_index}.idx'
            if os.path.exists(binary_file) and os.path.exists(index_file):
                mode = 'r'
            else:
                mode = 'w'

            self.cache.record = BinaryBlob(
                binary_file=binary_file,
                index_file=index_file,
                mode=mode
            )

        logger.info("initVars__: {}".format(self.getInfo()))

    @verbose
    async def enter__(self):
        self.tasks.init_connection = await reCreate(self.tasks.init_connection, self.init_connection__)
        logger.debug("enter__: {}".format(self.getInfo()))

    async def init_connection__(self):
        """
        connect to a server
        retry if not successful
        then start reading socket
        """
        try:
            try:
                self.cur_attempt += 1
                self.reader, self.writer = await asyncio.open_connection(
                    self.host_adr, self.port
                )

                logger.info(
                    "init_connection__: successfully connected to {} on port {}".format(
                        self.host_adr,
                        self.port
                    )
                )
                self.is_connected = True
                # reset the number of attempts
                self.cur_attempt = 0
                # start reading from socket
                self.tasks.read_socket = await reCreate(
                    self.tasks.read_socket,
                    self.read_socket__
                )
                # also start processing payload
                self.tasks.process_payload = await reCreate(
                    self.tasks.process_payload,
                    self.process_payload__
                )

            except Exception as e:
                logger.info("init_connection__: connect failed with", e)
                await self.reset_connection__()
                if self.cur_attempt <= self.max_retry:
                    logger.info(f"init_connection__: trying to reconnect within {self.retry_interval} secs")
                    await asyncio.sleep(self.retry_interval)
                    # schedule a task to reconnect
                    self.tasks.init_connection = await reSchedule(
                        self.init_connection__
                    )
                else:
                    await self.warn_and_exit(
                        'init_connection__',
                        'reaching max attempt',
                    )

        except asyncio.CancelledError:
            logger.info("init_connection__ : cancelling %s", self.getInfo())
        except Exception as e:
            await self.warn_and_exit('init_connection__', str(e))

    async def reset_connection__(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        self.is_connected = False


    @verbose
    async def exit__(self):
        await self.reset_connection__()
        self.tasks.read_socket = await delete(self.tasks.read_socket)
        self.tasks.write_socket = await delete(self.tasks.write_socket)
        self.tasks.init_connection = await delete(self.tasks.init_connection)
        self.tasks.process_payload = await delete(self.tasks.process_payload)
        self.tasks.monitor_event = await delete(self.tasks.monitor_event)
        logger.debug(f"{self.getInfo()} exit__: bye!")

    async def write_socket__(self, obj=None):
        """
        Send an object to the server
        """
        try:
            # handle the case when not connected yet
            if not self.is_connected:
                await asyncio.sleep(0.01)
                await self.write_socket__(obj)
                return

            if obj is not None:
                # convert to byte array
                self.write_buf = object_to_bytes(obj)
                self.write_start = 0
                # then reschedule to call this task again
                self.tasks.write_socket = await reSchedule(self.write_socket__)
            else:
                # this part is the actual writing part
                # we will write one blob of BLOCK_SIZE at a time
                # and allow returning the control to the main loop if needed
                if self.write_start < len(self.write_buf):
                    write_stop = min(len(self.write_buf), self.write_start + self.packet_size)
                    try:
                        self.writer.write(self.write_buf[self.write_start:write_stop])
                        await self.writer.drain()
                    except Exception as e:
                        await self.warn_and_exit('write_socket__', str(e))
                    else:
                        self.write_start = write_stop
                        self.tasks.write_socket = await reSchedule(self.write_socket__)

        except asyncio.CancelledError:
            logger.info("write_socket__ : cancelling %s", self.getInfo())
        except Exception as e:
            await self.warn_and_exit('write_socket__', str(e))

    async def read_socket__(self):
        """
        Read byte stream from the server
        """
        try:
            if self.read_socket_counter > 0:
                can_read = True
                self.read_socket_counter -= 1
            else:
                if self.received_payload.qsize() > self.max_queue_size:
                    can_read = False
                else:
                    can_read = True
                self.read_socket_counter = self.max_queue_size

            if can_read:
                try:
                    packet = await self.reader.read(self.packet_size)
                    if len(packet) > 0:
                        # all good!  keep on reading = reschedule this
                        await self.handle_read_packet__(packet)
                        # reschedule the task
                        self.tasks.read_socket = await reSchedule(self.read_socket__)
                    else:
                        await self.warn_and_exit(
                            'read_socket__',
                            'server has closed connection',
                        )

                except Exception as e:
                    await self.warn_and_exit(
                        'read_socket__',
                        str(e),
                    )
            else:
                await asyncio.sleep(0.001)
                self.tasks.read_socket = await reSchedule(self.read_socket__)

        except asyncio.CancelledError:
            self.print_warning(
                'read_socket__',
                'got canceled',
            )

        except Exception as e:
            await self.warn_and_exit('read_socket__', str(e))


    def reset_read_package_state__(self, clearbuf = False):
        self.left = INTERCOM_HEADER_LEN
        self.obj_len = 0
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
                self.obj_len = int.from_bytes(self.read_buf[0:INTERCOM_HEADER_LEN], "big")
                self.header = False # we got the header info (length)
                if len(self.read_buf) > INTERCOM_HEADER_LEN:
                    # sort out the remaining stuff
                    await self.handle_read_packet__(None)
        else:
            if len(self.read_buf) >= (INTERCOM_HEADER_LEN + self.obj_len):
                # correct amount of bytes have been obtained
                payload = bytes_to_object(
                    self.read_buf[INTERCOM_HEADER_LEN:INTERCOM_HEADER_LEN + self.obj_len]
                )
                if self.total_minibatch is None:
                    logger.info(f'received total minibatch: {payload}')
                    self.total_minibatch = payload
                    self.event_queue.put({'status': 'ready', 'total_minibatch': self.total_minibatch})

                    # start task to monitor event queue
                    self.tasks.monitor_event = await reCreate(
                        self.tasks.monitor_event,
                        self.monitor_event__
                    )
                else:
                    # put reconstructed objects into a queue
                    if self.start_time is None:
                        self.start_time = time.time()
                    self.fps_count += 1
                    if self.fps_count == 100:
                        duration = time.time() - self.start_time
                        latency = duration / 100
                        logger.info('it took {:.4f} seconds to reconstruct 1 minibatch'.format(latency))
                        self.fps_count = 0
                        self.start_time = time.time()

                    self.received_payload.put((self.minibatch_idx, payload))
                    self.minibatch_idx += 1
                    if self.minibatch_idx == self.total_minibatch:
                        # reset counter
                        self.minibatch_idx = 0
                        # put None to signal the end of epoch
                        self.received_payload.put(None)

                # also write confirmation back to the server
                await self.write_socket__('ok')

                # prepare state for next blob
                if len(self.read_buf) > (INTERCOM_HEADER_LEN + self.obj_len):
                    # there's some leftover here for the next blob..
                    self.read_buf = self.read_buf[INTERCOM_HEADER_LEN + self.obj_len:]
                    self.reset_read_package_state__()
                    # .. so let's handle that leftover
                    await self.handle_read_packet__(None)
                else:
                    # blob ends exactly
                    self.reset_read_package_state__(clearbuf=True)

    def is_data_from_server(self):
        if self.cache is None:
            return True
        if self.cache.current_epoch == 0:
            self.cache.read_indices = shuffle_indices(0, self.total_minibatch, self.nearby_shuffle)
            if self.cache.record.mode() == 'read':
                return False
            else:
                return True
        else:
            if self.cache.current_epoch % self.cache.update_frequency == 0:
                return True
            else:
                return False

    @verbose
    async def process_payload__(self):
        """
        recurring task to process the received_payload queue
        """
        try:
            # check whether we need to read from cache or process the payload queue
            from_server = self.is_data_from_server()
            if from_server:
                # if data is reconstructed from payload
                try:
                    payload = self.received_payload.get(block=False)
                    if payload is None:
                        # end of epoch, also put None to signal the parent process
                        self.data_queue.put(None)
                        # we also need to close the record and open again in read me
                        if self.cache is not None:
                            self.cache.record.close()
                            self.cache.record = BinaryBlob(
                                self.cache.record.binary_file(),
                                self.cache.record.index_file(),
                                mode='r',
                            )
                    else:
                        # payload contains the index of current minibatch too
                        idx, payload = payload
                        # otherwise, simply put the minibatch to data_queue
                        self.data_queue.put(payload)
                        # if caching, then write the minibatch
                        if self.cache is not None:
                            self.cache.record.write_index(idx, payload)

                except queue.Empty:
                    # we read without blocking so there's a chance that we have
                    # nothing to read, sleep a bit
                    await asyncio.sleep(0.001)
            else:
                # data is read from cache, this only happens when caching is enbaled
                idx = self.cache.read_indices.pop(0)
                minibatch = self.cache.record.read_index(idx)
                self.data_queue.put((1, minibatch))
                if len(self.cache.read_indices) == 0:
                    self.cache.read_indices = shuffle_indices(
                        0,
                        self.total_minibatch,
                        self.nearby_shuffle
                    )

            self.tasks.process_payload = await reSchedule(self.process_payload__)

        except asyncio.CancelledError:
            self.print_warning(
                'process_payload__',
                'got canceled',
            )
        except Exception as e:
            await self.warn_and_exit('process_payload__', str(e))


    @verbose
    async def monitor_event__(self):
        try:
            if not self.event_queue.empty():
                try:
                    event = self.event_queue.get(block=False)
                    if isinstance(event, dict):
                        self.event_queue.put(event)
                        await asyncio.sleep(5)
                    elif event == 'close':
                        # receive closing signal
                        logger.info('monitor_event__: receive closing signal')
                        await self.stop()
                except queue.Empty:
                    pass
            await asyncio.sleep(0.01)
            self.tasks.monitor_event = await reSchedule(self.monitor_event__)

        except asyncio.CancelledError:
            self.print_warning(
                'monitor_event__',
                'got canceled',
            )
        except Exception as e:
            await self.warn_and_exit('monitor_event__', str(e))

    @verbose
    async def warn_and_exit(self, function_name, message):
        self.event_queue.put({'status': 'failed'})
        await super().warn_and_exit(function_name, message)
