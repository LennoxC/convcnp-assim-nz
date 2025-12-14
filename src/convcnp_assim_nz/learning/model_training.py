import torch.optim as optim
import numpy as np
import lab as B
from typing import Dict, List
from tqdm import tqdm
import random

# TODO: Investigate using torch.amp.GradScaler for mixed precision training

# TODO: this doesn't use batching so is quite inefficient
def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)


"""
Train for one epoch over many tasks, each with potentially different numbers of target points.
Since ConvCNP does not support batching over different input sizes, we have to do this one at a time.
This is inefficient, a better approach can be using batch_data_by_num_stations to group tasks with the same number of stations together,
and then use deepsensor.train.train_epoch to train over these batches.

This could be useful for fine-tuning a model which has already been trained with a large dataset using batching.
The dataset with batching can be generated with `batch_data_by_num_stations`.

As of experiment 2 - this is not used anywhere. I prefer the default deepsensor training loop with batching as this is much faster.
"""

def train_epoch_many_targets(
    model,
    tasks,
    opt,
    batch_size: int = 1
) -> List[float]:

    tasks = np.random.permutation(tasks)

    if batch_size is not None:
        n_batches = len(tasks) // batch_size  # Note that this will drop the remainder
    else:
        n_batches = len(tasks)

    batch_losses = []
    for batch_i in tqdm(range(n_batches)):

        # zero the gradient at the start of the batch
        opt.zero_grad()

        losses_within_batch = []
        for b in range(batch_size):
            # ConvCNP doesn't support multi-input batching, so we have to do this one at a time
            task = tasks[batch_i * batch_size + b]
            task_loss = model.loss_fn(task, normalise=True)
            losses_within_batch.append(task_loss)

        mean_batch_loss = B.mean(B.stack(*losses_within_batch))
        mean_batch_loss.backward()
        opt.step()

        batch_losses.append(mean_batch_loss.detach().cpu().numpy())
        
    return batch_losses

def batch_data_by_num_stations(tasks, batch_size=None):
        # if batch_size == None, return a dict in which each key value pair is a 
        # list of *all of the tasks* with the same number of stations
        batched_tasks = {}
        for task in tasks:
            num_stations = task['X_t'][0].shape[1]
            if f'{num_stations}' not in batched_tasks.keys():
                batched_tasks[f'{num_stations}'] = [task]
            else:
                batched_tasks[f'{num_stations}'].append(task)

        # if batch_size is not None, return a dict in which each key value pair is a
        # list of tasks with the same number of stations, but with batch_size number of tasks
        # e.g. if batch_size = 4 but there are 10 tasks with 100 stations, then there will be 100 keys:
        # '100_0' with 4 tasks, '100_1' with 4 tasks, '100_2' with 2 tasks
                
        # reason for doing it like this: if we set a large batch_size, e.g. 16, and there are 
        # only 10 tasks with 100 stations, then the deepsensor train_epoch function will 
        # ignore these tasks 
        if batch_size is not None:
            batched_tasks_copy = batched_tasks.copy()
            batched_tasks = {}
            for num_stations in batched_tasks_copy.keys():
                number_tasks_in_batch = len(batched_tasks_copy[f'{num_stations}'])
                for idx, i in enumerate(range(0, number_tasks_in_batch, batch_size)):
                    batched_tasks[f'{num_stations}_{idx}'] = batched_tasks_copy[f'{num_stations}'][i:i+batch_size]

        return batched_tasks