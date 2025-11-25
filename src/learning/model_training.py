import torch.optim as optim
import deepsensor
from deepsensor.data.task import Task, concat_tasks
from deepsensor.model.convnp import ConvNP
import numpy as np
import lab as B
from typing import List
from tqdm import tqdm

# TODO: Investigate using torch.amp.GradScaler for mixed precision training

def train_epoch_many_targets(
    model: ConvNP,
    tasks: List[Task],
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
