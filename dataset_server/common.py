"""
common.py: common tools
-----------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""
from __future__ import annotations
import dill
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
import random
from task_thread import (
    reCreate,
    reSchedule,
    delete,
    verbose,
    signals
)
import copy
from task_thread import TaskThread as BaseTaskThread


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


def shuffle_indices(start_idx, stop_idx, nearby_shuffle):
    indices = list(range(start_idx, stop_idx))
    if nearby_shuffle > 0:
        i = random.choice(indices[1:-1])
        indices_ = indices[i:] + indices[:i]
        indices = []
        start_index = 0
        N = len(indices_)
        while len(indices) < N:
            stop_index = min(N, start_index + nearby_shuffle)
            sub_indices = indices_[start_index: stop_index]
            random.shuffle(sub_indices)
            indices.extend(sub_indices)
            start_index = stop_index
    else:
        random.shuffle(indices)

    return indices


class BinaryBlob(TorchDataset):
    """
    abstraction for binary blob storage
    taken from mlproject.data (https://github.com/viebboy/mlproject)
    """
    def __init__(self, binary_file: str, index_file: str, mode='r'):
        assert mode in ['r', 'w']
        self._mode = 'write' if mode == 'w' else 'read'
        self._binary_file = binary_file
        self._index_file = index_file

        if mode == 'w':
            # writing mode
            self._fid = open(binary_file, 'wb')
            self._idx_fid = open(index_file, 'w')
            self._indices = set()
        else:
            assert os.path.exists(binary_file)
            assert os.path.exists(index_file)

            # read index file
            with open(index_file, 'r') as fid:
                content = fid.read().split('\n')[:-1]

            self._index_content = {}
            self._indices = set()
            for row in content:
                sample_idx, byte_pos, byte_length, need_conversion = row.split(',')
                self._index_content[int(sample_idx)] = (
                    int(byte_pos),
                    int(byte_length),
                    bool(int(need_conversion)),
                )
                self._indices.add(int(sample_idx))

            # open binary file
            self._fid = open(binary_file, 'rb')
            self._fid.seek(0, 0)
            self._idx_fid = None

            # sorted indices
            self._sorted_indices = list(self._indices)
            self._sorted_indices.sort()

    def mode(self):
        return self._mode

    def binary_file(self):
        return self._binary_file

    def index_file(self):
        return self._index_file

    def __getitem__(self, i: int):
        if self._mode == 'write':
            raise RuntimeError('__getitem__ is not supported when BinaryBlob is opened in write mode')

        if idx >= len(self):
            raise RuntimeError(f'index {i} is out of range: [0 - {len(self)})')
        idx = self._sorted_indices[i]
        return self.read_index(idx)

    def __len__(self):
        if self._mode == 'write':
            raise RuntimeError('__len__ is not supported when BinaryBlob is opened in write mode')
        return len(self._sorted_indices)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_index(self, index: int, content):
        assert isinstance(index, int)
        if self._mode == 'write':
            # allow writing
            try:
                # check if index existence
                if index in self._indices:
                    raise RuntimeError(f'Given index={index} has been occuppied. Cannot write')

                # convert to byte string
                if not isinstance(content, bytes):
                    content = dill.dumps(content)
                    # flag to mark whether serialization/deserialization is
                    # needed
                    converted = 1
                else:
                    converted = 0

                # log position before writing
                current_pos = self._fid.tell()
                # write byte string
                self._fid.write(content)
                # write metadata information
                self._idx_fid.write(f'{index},{current_pos},{len(content)},{converted}\n')

                # keep track of index
                self._indices.add(index)

            except Exception as error:
                self.close()
                raise error
        else:
            # raise error
            self.close()
            raise RuntimeError('BinaryBlob was opened in reading mode. No writing allowed')

    def read_index(self, index: int):
        assert isinstance(index, int)
        assert index >= 0

        if self._mode == 'read':
            if index not in self._indices:
                self.close()
                raise RuntimeError(f'Given index={index} does not exist in BinaryBlob')

            # pos is the starting position we need to seek
            target_pos, length, need_conversion = self._index_content[index]
            # we need to compute seek parameter
            delta = target_pos - self._fid.tell()
            # seek to target_pos
            self._fid.seek(delta, 1)

            # read `length` number of bytes
            item = self._fid.read(length)
            # deserialize if needed
            if need_conversion:
                item = dill.loads(item)
            return item
        else:
            self.close()
            raise RuntimeError('BinaryBlob was opened in writing mode. No reading allowed')

    def close(self):
        if self._fid is not None:
            self._fid.close()
        if self._idx_fid is not None:
            self._idx_fid.close()


class Property:
    """
    property placeholder
    """
    def __init__(self):
        pass


class TaskThread(BaseTaskThread):
    """
    A modified base TaskThread class that implements some functionalities
    """

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        super().__init__(parent=parent)

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
