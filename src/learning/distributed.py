import torch.distributed as dist
import torch 
import os

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():
    return get_rank() == 0 or os.environ.get("USING_DDP", "False") == "False"

def init_distributed():
    """Initialize distributed training environment."""
    dist_url = "env://" # default

    # these are set by torch.distributed.launch
    RANK = int(os.environ.get("RANK")) # rank of process in the network
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE")) # total number of processes
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK")) # rank of process on the local machine
    
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=WORLD_SIZE,
        rank=RANK,
    )

    torch.cuda.set_device(LOCAL_RANK)

    dist.barrier()  # synchronize all processes


import math
import random
from typing import Callable, List, Iterable, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

class LoaderWrapper:
    """
    Wrap a PyTorch DataLoader so that iterating over LoaderWrapper yields
    pre-batched task objects, but does NOT materialize them all at once.
    """
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        # Just yield batches one by one
        for batch in self.loader:
            yield batch

    def __len__(self):
        return len(self.loader)

# -------------------------
# 1) Simple Dataset wrapper
# -------------------------
class TaskDataset(Dataset):
    """Wrap a python list of `task` objects for use with DataLoader."""
    def __init__(self, tasks: List[Any]):
        self.tasks = list(tasks)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

# -------------------------------------------------------
# 2) Create batches grouped by a length function (offline)
# -------------------------------------------------------
def make_batches_grouped_by_length(
    tasks: Iterable[Any],
    batch_size: int,
    length_fn: Callable[[Any], int],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    Return a list-of-batches where each batch is a list of dataset indices.
    Batches only contain tasks with the same `length_fn(task)` value.

    - tasks: iterable of tasks (same order as dataset)
    - length_fn: function(task) -> int (the value we group by, e.g. num_stations)
    - returns: list of batches (each batch is list of integer indices)
    """
    tasks = list(tasks)
    # group indices by length
    groups = {}
    for i, t in enumerate(tasks):
        k = length_fn(t)
        groups.setdefault(k, []).append(i)

    # optional shuffle groups' indices
    rng = random.Random(seed)
    batches = []
    for k, idxs in groups.items():
        if shuffle:
            rng.shuffle(idxs)
        # chunk into batches preserving same-length property
        for i in range(0, len(idxs), batch_size):
            batch = idxs[i : i + batch_size]
            # optionally drop last incomplete batch? keep it by default
            batches.append(batch)

    # shuffle the order of batches themselves (we kept group-internal shuffling)
    if shuffle:
        rng.shuffle(batches)
    return batches

# -------------------------------------------------------
# 3) A small BatchSampler-like wrapper that is DDP-aware
# -------------------------------------------------------
class DistributedBatchSampler:
    """
    Given a precomputed list-of-batches (each batch is list[int] indices),
    yield only the batches that belong to this rank.

    Behavior:
      - batches: list of batches
      - drop_last: if True, drop final uneven batches across ranks so each rank has the same count
      - pad: if True and drop_last False, duplicate batches to make batches divisible by world_size
      - epoch: you can call set_epoch(epoch) to reshuffle deterministically earlier (if you prefer)
    """
    def __init__(
        self,
        batches: List[List[int]],
        drop_last: bool = False,
        pad: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self._batches = list(batches)
        self.drop_last = drop_last
        self.pad = pad
        # get world info from torch.distributed if not supplied
        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
            else:
                self.world_size = 1
                self.rank = 0
        else:
            self.world_size = world_size
            self.rank = rank
        self.epoch = 0

    def __len__(self):
        # how many batches this sampler will yield for this rank
        total = len(self._batches)
        if self.drop_last:
            total = (total // self.world_size) * self.world_size
        else:
            # pad to multiple if requested, else ceil-divide
            if self.pad:
                total = math.ceil(total / self.world_size) * self.world_size
            # else we allow uneven distribution
        return total // self.world_size

    def set_epoch(self, epoch: int):
        """optional: call this each training epoch to reseed deterministic shuffles."""
        self.epoch = int(epoch)

    def __iter__(self):
        batches = list(self._batches)  # copy
        # If pad requested, duplicate first batches until divisible
        total = len(batches)
        if self.pad and self.world_size > 1:
            target = math.ceil(total / self.world_size) * self.world_size
            # duplicate from start if needed
            i = 0
            while len(batches) < target:
                batches.append(batches[i % total])
                i += 1
            total = len(batches)

        if self.drop_last and self.world_size > 1:
            total = (total // self.world_size) * self.world_size
            batches = batches[:total]

        # now slice for this rank
        out_batches = batches[self.rank::self.world_size]
        for b in out_batches:
            yield b

# -----------------------
# 4) collate function
# -----------------------
def make_concat_collate_fn(concat_fn: Callable[[List[Any]], Any]):
    """
    Returns a collate_fn that takes list-of-task (the dataset items)
    and returns the concatenated/merged batch using concat_fn.
    concat_fn should accept a list of tasks and return a single batched task.
    """
    def collate_fn(list_of_tasks: List[Any]):
        # rely on user's concat_tasks to handle merging
        return concat_fn(list_of_tasks)
    return collate_fn

def length_fn(task):
    return task['X_t'][0].shape[1] # number of stations

# -----------------------
# 5) Putting it together
# -----------------------
def build_ddp_loader_from_tasks(
    tasks: List[Any],
    batch_size: int,
    length_fn: Callable[[Any], int],
    concat_fn: Callable[[List[Any]], Any],
    *,
    shuffle_batches: bool = True,
    seed: int = 1234,
    drop_last: bool = False,
    pad: bool = False,
    num_workers: int = 2,
    pin_memory: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Construct a DataLoader suitable for DDP training.
    - tasks: a list of task objects
    - length_fn: function(task) -> grouping key (e.g. number of stations)
    - concat_fn: function(list_of_tasks) -> batched_task (your concat_tasks)
    """
    dataset = TaskDataset(tasks)
    batches = make_batches_grouped_by_length(
        tasks, batch_size=batch_size, length_fn=length_fn, shuffle=shuffle_batches, seed=seed
    )
    # create distributed sampler wrapper
    dbs = DistributedBatchSampler(
        batches=batches,
        drop_last=drop_last,
        pad=pad
    )
    collate_fn = make_concat_collate_fn(concat_fn)

    # DataLoader: we pass batch_sampler (so do NOT pass shuffle or sampler)
    loader = DataLoader(
        dataset,
        batch_sampler=dbs,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dbs  # return dbs so caller can call set_epoch(epoch) each epoch
