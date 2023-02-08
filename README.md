# Dataset Server - Bootstrapping your datasets/dataloaders in other processes

Overcoming expensive data preprocessing with async datasets/dataloaders. 

## Installation

Clone this repo, install dependencies in `requirements.txt` and run `pip install -e .` to install the package in development mode. 

## Usage

Examples can be found under `examples`. 

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
from dataset_server import AsyncDataLoader

# assuming SuitableDataset is constructed from keyword arguments like the one above
# and argument values used to construct it are the following
params = {
    'param1': #some value for 1st positional argument,
    'param2': #some value for 2nd positional argument,
}

async_dataloader = AsyncDataLoader(
    dataset_class=SuitableDataset, #pass the dataset class here, not the object
    dataset_params=params, # keyword arguments used to construct dataset object here 
    batch_size=128, # specify the batch size of the data loader here
    nb_servers=1, # specify the number of processes (TCP servers) to serve the dataset. More servers consume more CPUs but also faster
    shuffle=True, # whether to perform shuffling
)

# then use this data loader like any other loaders:
for samples in async_dataloader:
    # perform processing here with the samples
```
