import torch.optim as optim
import numpy as np
import lab as B
from typing import Dict, List
from tqdm import tqdm
import random
#from deepsensor.data.task import concat_tasks
import math

# TODO: Investigate using torch.amp.GradScaler for mixed precision training

def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)

def compute_val_loss_pickled(model, val_task_dir, batch_size: int = None, epoch: int = None, repeat_sampling: int = None):
    import pickle
    import os
    import torch

    val_losses = []

    def val_step(tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        with torch.no_grad():
            task_losses = []

            for task in tasks:
                task_losses.append(model.loss_fn(task, fix_noise=None, normalise=True))

            mean_batch_loss = B.mean(B.stack(*task_losses))
        
        return mean_batch_loss.detach().cpu().numpy()

    # list all tasks in val_task_dir
    if epoch is not None and repeat_sampling is not None:
        file_ext = epoch % repeat_sampling
        val_task_files = [f for f in os.listdir(val_task_dir) if f.endswith(f'_{file_ext}.pkl')]
    else:
        val_task_files = [f for f in os.listdir(val_task_dir) if f.endswith('.pkl')]

    if batch_size is not None:
        n_batches = len(val_task_files) // batch_size  # Note that this will drop the remainder
    else:
        n_batches = len(val_task_files)

    for batch_i in tqdm(range(n_batches)):

        # load the first batch_size tasks from val_task_files
        tasks = []
        for b in range(batch_size):
            task_file = val_task_files[batch_i * batch_size + b]
            with open(os.path.join(val_task_dir, task_file), 'rb') as f:
                task = pickle.load(f)
                tasks.append(task)

        if batch_size is not None:
            task = concat_tasks_custom(tasks)
        else:
            task = tasks[batch_i]
        
        batch_loss = val_step(task)

        val_losses.append(batch_loss)
        
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

def train_epoch_pickled(
    model,
    train_task_dir,
    opt,
    batch_size: int = 1,
    epoch: int = None,
    repeat_sampling: int = None,
    use_grad_clip: bool = False, 
    grad_clip_value: float = 0.0,
    grad_accum_steps: int = 1
    ):

    import pickle
    import os

    if use_grad_clip:
        from torch.nn.utils import clip_grad_norm_

    def train_step(tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        opt.zero_grad()
        task_losses = []

        for task in tasks:
            task_losses.append(model.loss_fn(task, fix_noise=None, normalise=True))

        mean_batch_loss = B.mean(B.stack(*task_losses))
        (mean_batch_loss / grad_accum_steps).backward()

        if (batch_i + 1) % grad_accum_steps == 0:
            if use_grad_clip:
                clip_grad_norm_(model.model.parameters(), grad_clip_value)

            opt.step()
            opt.zero_grad()

        return mean_batch_loss.detach().cpu().numpy()

    # list all tasks in train_task_dir
    # when epoch is specified, this indicates we are using each timestep only once per epoch.
    #    As multiple tasks were generated per timestep, we filter the task files to only those
    #    corresponding to the current epoch. This helps mitigate immediate overfitting when using
    #    small datasets.
    if epoch is not None and repeat_sampling is not None:
        file_ext = epoch % repeat_sampling
        train_task_files = [f for f in os.listdir(train_task_dir) if f.endswith(f'_{file_ext}.pkl')]
    else:
        train_task_files = [f for f in os.listdir(train_task_dir) if f.endswith('.pkl')]
    
    train_task_files = np.random.permutation(train_task_files)

    if batch_size is not None:
        n_batches = len(train_task_files) // batch_size  # Note that this will drop the remainder
    else:
        n_batches = len(train_task_files)

    batch_losses = []
    for batch_i in tqdm(range(n_batches)):

        # load the first batch_size tasks from train_task_files
        tasks = []
        for b in range(batch_size):
            task_file = train_task_files[batch_i * batch_size + b]
            with open(os.path.join(train_task_dir, task_file), 'rb') as f:
                task = pickle.load(f)
                tasks.append(task)

        if batch_size is not None:
            task = concat_tasks_custom(tasks)
        else:
            task = tasks[batch_i]
        
        batch_loss = train_step(task)

        batch_losses.append(batch_loss)
        
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

# the default deepsensor concat_tasks function does not handle correct batching 
# of Y_t_aux. This custom function version concatenates Y_t_aux correctly.

from deepsensor.data.task import Task
import deepsensor
import copy

def concat_tasks_custom(tasks: List[Task], multiple: int = 1) -> Task:
    """Concatenate a list of tasks into a single task containing multiple batches.

    ..

    Todo:
        - Consider moving to ``nps.py`` as this leverages ``neuralprocesses``
          functionality.
        - Raise error if ``aux_t`` values passed (not supported I don't think)

    Args:
        tasks (List[:class:`deepsensor.data.task.Task`:]):
            List of tasks to concatenate into a single task.
        multiple (int, optional):
            Contexts are padded to the smallest multiple of this number that is
            greater than the number of contexts in each task. Defaults to 1
            (padded to the largest number of contexts in the tasks). Setting
            to a larger number will increase the amount of padding but decrease
            the range of tensor shapes presented to the model, which simplifies
            the computational graph in graph mode.

    Returns:
        :class:`~.data.task.Task`: Task containing multiple batches.

    Raises:
        ValueError:
            If the tasks have different numbers of target sets.
        ValueError:
            If the tasks have different numbers of targets.
        ValueError:
            If the tasks have different types of target sets (gridded/
            non-gridded).
    """
    if len(tasks) == 1:
        return tasks[0]

    for i, task in enumerate(tasks):
        if "numpy_mask" in task["ops"] or "nps_mask" in task["ops"]:
            raise ValueError(
                "Cannot concatenate tasks that have had NaNs masked. "
                "Masking will be applied automatically after concatenation."
            )
        if "target_nans_removed" not in task["ops"]:
            task = task.remove_target_nans()
        if "batch_dim" not in task["ops"]:
            task = task.add_batch_dim()
        if "float32" not in task["ops"]:
            task = task.cast_to_float32()
        tasks[i] = task

    # Assert number of target sets equal
    n_target_sets = [len(task["Y_t"]) for task in tasks]
    if not all([n == n_target_sets[0] for n in n_target_sets]):
        raise ValueError(
            f"All tasks must have the same number of target sets to concatenate: got {n_target_sets}. "
        )
    n_target_sets = n_target_sets[0]

    for target_set_i in range(n_target_sets):
        # Raise error if target sets have different numbers of targets across tasks
        n_target_obs = [task["Y_t"][target_set_i].size for task in tasks]
        if not all([n == n_target_obs[0] for n in n_target_obs]):
            raise ValueError(
                f"All tasks must have the same number of targets to concatenate: got {n_target_obs}. "
                "To train with Task batches containing differing numbers of targets, "
                "run the model individually over each task and average the losses."
            )

        # Raise error if target sets are different types (gridded/non-gridded) across tasks
        if isinstance(tasks[0]["X_t"][target_set_i], tuple):
            for task in tasks:
                if not isinstance(task["X_t"][target_set_i], tuple):
                    raise ValueError(
                        "All tasks must have the same type of target set (gridded or non-gridded) "
                        f"to concatenate. For target set {target_set_i}, got {type(task['X_t'][target_set_i])}."
                    )

    # For each task, store list of tuples of (x_c, y_c) (one tuple per context set)
    contexts = []
    for i, task in enumerate(tasks):
        contexts_i = list(zip(task["X_c"], task["Y_c"]))
        contexts.append(contexts_i)

    # List of tuples of merged (x_c, y_c) along batch dim with padding
    # (up to the smallest multiple of `multiple` greater than the number of contexts in each task)
    merged_context = [
        deepsensor.backend.nps.merge_contexts(
            *[context_set for context_set in contexts_i], multiple=multiple
        )
        for contexts_i in zip(*contexts)
    ]

    merged_task = copy.deepcopy(tasks[0])

    # Convert list of tuples of (x_c, y_c) to list of x_c and list of y_c
    merged_task["X_c"] = [c[0] for c in merged_context]
    merged_task["Y_c"] = [c[1] for c in merged_context]
    merged_task["Y_t_aux"] = B.concat(*[t["Y_t_aux"] for t in tasks]) # concatenate aux targets for all targets

    # This assumes that all tasks have the same number of targets
    for i in range(n_target_sets):
        if isinstance(tasks[0]["X_t"][i], tuple):
            # Target set is gridded with tuple of coords for `X_t`
            merged_task["X_t"][i] = (
                B.concat(*[t["X_t"][i][0] for t in tasks], axis=0),
                B.concat(*[t["X_t"][i][1] for t in tasks], axis=0),
            )
        else:
            # Target set is off-the-grid with tensor for `X_t`
            merged_task["X_t"][i] = B.concat(*[t["X_t"][i] for t in tasks], axis=0)
        merged_task["Y_t"][i] = B.concat(*[t["Y_t"][i] for t in tasks], axis=0)

    merged_task["time"] = [t["time"] for t in tasks]

    merged_task = Task(merged_task)

    # Apply masking
    merged_task = merged_task.mask_nans_numpy()
    merged_task = merged_task.mask_nans_nps()

    return merged_task

def return_sample_predictions(era5_date_ds, h8_date_ds, nzra_date_ds, nzra_ds, stations_date_df, ds_aux_processed, ds_aux_coarse_processed, val_date, data_processor, epoch, model, target, padding = 200, subtitle_text="Placeholder subtitle text"):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from deepsensor.data.loader import TaskLoader
    from convcnp_assim_nz.utils.variables.coord_names import LATITUDE, LONGITUDE

    era5_date_processed, h8_date_processed, nzra_processed, stations_date_processed = data_processor([era5_date_ds, h8_date_ds, nzra_date_ds, stations_date_df])

    temp_task_loader = TaskLoader(
        context = [h8_date_processed, stations_date_processed, era5_date_processed, ds_aux_coarse_processed], 
        target = nzra_processed,
        aux_at_targets = ds_aux_processed)
    
    # the task for inference/prediction
    task = temp_task_loader(val_date, context_sampling=["all", "all", "all", "all"], target_sampling=["all"])

    pred = model.predict(task, X_t=nzra_ds[[LATITUDE, LONGITUDE]])

    # Number of annotated grid points
    N_ANN = 5

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Epoch {epoch} Predictions for {val_date}', fontsize=18)
    fig.text(0.5, 0.94, subtitle_text, ha='center', fontsize=12, color='gray')

    # --- DATA EXTRACTION ---
    truth_field = nzra_ds[target].sel(time=val_date).values
    pred_mean  = pred[target]["mean"].isel(time=0).values
    pred_std   = pred[target]["std"].isel(time=0).values

    Ny, Nx = truth_field.shape

    # add padding to the annotated points
    Ny = Ny - padding//2
    Nx = Nx - padding//2

    # Shared colour scale across truth + prediction
    vmin = min(truth_field.min(), pred_mean.min())
    vmax = max(truth_field.max(), pred_mean.max())

    # Annotated grid points
    ys = np.linspace(padding//2, Ny - 1, N_ANN, dtype=int)
    xs = np.linspace(padding//2, Nx - 1, N_ANN, dtype=int)
    annot_points = [(y, x) for y in ys for x in xs]

    # --- TOP LEFT: NZRA temperature ---
    im0 = ax[0, 0].imshow(truth_field, origin='lower', vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(f"NZRA {target}")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.7)
    for y, x in annot_points:
        ax[0, 0].text(x, y, f"{truth_field[y,x]:.1f}", fontsize=7,
                    color='white', ha='center', va='center')

    # --- TOP RIGHT: stations ---
    ax[0, 1].set_aspect('equal', adjustable='box')

    m = Basemap(
        projection='merc',
        llcrnrlat=nzra_date_ds[LATITUDE].min().item(),
        urcrnrlat=nzra_date_ds[LATITUDE].max().item(),
        llcrnrlon=nzra_date_ds[LONGITUDE].min().item(),
        urcrnrlon=nzra_date_ds[LONGITUDE].max().item(),
        resolution='i',
        ax=ax[0, 1]
    )

    m.drawcoastlines()
    m.drawcountries()

    df_day = stations_date_df.reset_index().loc[
        lambda df: df['time'] == val_date
    ]

    x, y = m(df_day['lon'].values, df_day['lat'].values)
    sc = m.scatter(x, y,
                c=df_day[target].values,
                cmap='coolwarm',
                marker='o',
                edgecolor='k',
                s=80)

    ax[0, 1].set_title("Station Observations")
    fig.colorbar(sc, ax=ax[0, 1], shrink=0.7)

    # --- BOTTOM LEFT: PRED MEAN ---
    im2 = ax[1, 0].imshow(pred_mean, origin='lower', vmin=vmin, vmax=vmax)
    ax[1, 0].set_title(f"ConvCNP {target} prediction (mean)")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.7)
    for y, x in annot_points:
        ax[1, 0].text(x, y, f"{pred_mean[y,x]:.1f}", fontsize=7,
                    color='white', ha='center', va='center')

    # --- BOTTOM RIGHT: PRED STD ---
    im3 = ax[1, 1].imshow(pred_std, origin='lower')
    ax[1, 1].set_title("ConvCNP Standard Deviation")
    fig.colorbar(im3, ax=ax[1, 1], shrink=0.7)
    for y, x in annot_points:
        ax[1, 1].text(x, y, f"{pred_std[y,x]:.2f}", fontsize=6,
                    color='white', ha='center', va='center')

    plt.tight_layout()

    return fig