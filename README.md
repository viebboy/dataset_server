# Dataset Server - Bootstrapping your datasets/dataloaders in other processes

Overcoming expensive data preprocessing with async datasets/dataloaders. 

This allows your data loading to run in separate processes while your model or analysis is running.

In addition, when getting item from your dataset is expensive, you can also enable caching (saving the entire dataset in binary files for quicker read) and specify the frequency of cache update for every K epochs (in case your dataset performs random augmentation). 

When caching is not an option because saving the entire dataset requires too much additional disk space, you can turn on the rotation feature, which basically caches a small portion of your dataset and allows a sample to be generated more than once. 

Rotation can be done either on memory or on disk space. 


## Installation

Clone this repo, install dependencies in `requirements.txt` and run `pip install -e .` to install the package in development mode. 

## Usage

Basically, to create an async data loader, we need a dataset class that is constructed with only keyword arguments. 

That is, the dataset must be created like this `dataset = YourDataset(**kwargs)`. 

If your dataset is only constructed with positional arguments, you can bootstrap it to receive only keyword arguments like this:

```python
# assuming that YourDataset is constructed like this
# dataset = YourDataset(param1_value, param2_value)
# we could create another dataset class that inherits from YourDataset and constructs only from keyword arguments

class SuitableDataset(YourDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['param1'], kwargs['param2'])
```

After having a dataset class that is constructed only from keyword arguments, we could create an async data loader like this:

```python
from dataset_server import DataLoader

# assuming SuitableDataset is constructed from keyword arguments like the one above
# and argument values used to construct it are the following
params = {
    'param1': #some value for 1st positional argument,
    'param2': #some value for 2nd positional argument,
}

async_dataloader = DataLoader(
    dataset_class=SuitableDataset, #pass the dataset class here, not the object
    dataset_params=params, # keyword arguments used to construct dataset object here 
    batch_size=128, # specify the batch size of the data loader here
    nb_worker=1, # specify the number of worker processes to load from the dataset
    shuffle=True, # whether to perform shuffling
)

# then use this data loader like any other loaders:
for samples in async_dataloader:
    # perform processing here with the samples
```

## dataset_server.DataLoader in-depth 

The signature of `dataset_server.DataLoader` looks like this:

```python

DataLoader(
    self,
    dataset_class,
    dataset_params,
    batch_size,
    nb_worker,
    max_minibatch_length=None, 
    max_queue_size=20,
    prefetch_time=None,
    shuffle=False,
    device=None,
    pin_memory=False,
    gpu_indices=None,
    nearby_shuffle=0,
    cache_setting=None,
    rotation_setting=None,
    use_threading=False,
    collate_fn=None
)
```

with:

- `dataset_class`: refers to the name of your dataset class
- `dataset_params` (dict): is a dictionary that contains parameters of your dataset class.
  Your dataset should be implemented as keyword-based construction.
- `batch_size` (int): the size of minibatch
- `nb_worker` (int): the number of worker processes to read from dataset
- `max_minibatch_length` (int, default to None): the maximum number of bytes when representing your minibatch. 
  This should be left to None as in default so that the data loader will try to infer this value.
- `max_queue_size` (int, default to 20): the size of the queue in each worker.
  Increasing this value increases memory requirement.
- `prefetch_time` (int, default to None): if specified, the loader will wait for a given amount of seconds before completing the construction.
- `shuffle` (bool, default to None): whether to perform data shuffle.
- `device` (default to None): this is pytorch-related device. If specified, dataloader will move data to device
- `pin_memory` (bool, default to False): this is pytorch-related feature
- `gpu_indices` (list, default to None): if specified, this list should have length equal the number of workers.
  If your dataset uses GPU, you can optionally specify the GPU index that each worker should use.
- `nearby_shuffle` (int, default to 0): if larger than 0, the data loader will not generated samples from indices that are too far away (within the given number).
  This feature is useful if your dataset saves data in contiguous disk segment and accessing indices from far away can be costly (because we need to seek for too many bytes).
- `cache_setting` (dict, default to None): if specified, caching will be enabled. More on this later. 
- `rotation_setting` (dict, default to None): if specified, rotation will be enabled. More on this later.
- `use_threading` (bool, default to False): whether to use another thread to communicate with worker processes. 
  Use threading could improve latency if your training or analysis step takes a long time compared to communication with other workers. 
  The benefit of threading also depends on particular systems and use-cases.
- `collate_fn`: (callable, default to None): if specified, this function should takes a list of samples and perform sample collation. 
  The default collation is simply tensor (support numpy and torch) concatenation. 

