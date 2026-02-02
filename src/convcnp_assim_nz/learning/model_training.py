import numpy as np
import lab as B
from typing import List
from tqdm import tqdm
import pickle
import os
import torch
from matplotlib.colors import CenteredNorm
import matplotlib

# TODO: Investigate using torch.amp.GradScaler for mixed precision training

def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)

def compute_val_loss_pickled(
            model, 
            val_task_dir, 
            batch_size: int = None, 
            epoch: int = None, 
            fix_noise: float = None,
            n_subsample_targets: int = None,
        ):

    val_losses = []

    def val_step(tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        with torch.no_grad():
            task_losses = []

            for task in tasks:
                task_losses.append(model.loss_fn(task, edge_margin=(model.model.receptive_field/2), fix_noise=fix_noise, normalise=True))

            mean_batch_loss = B.mean(B.stack(*task_losses))
        
        return mean_batch_loss.detach().cpu().numpy()

    # list all tasks in val_task_dir
    val_task_files = [f for f in os.listdir(val_task_dir) if f.endswith('.pkl')]

    # limit for testing
    #val_task_files = val_task_files[:10*batch_size]

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

                if n_subsample_targets is not None:
                    X_t_new, Y_t_new, Y_t_aux_new = subsample_targets(task['X_t'], task['Y_t'], task['Y_t_aux'], n_subsample_targets, seed=epoch, replace=False)

                    task['X_t'] = X_t_new
                    task['Y_t'] = Y_t_new
                    task['Y_t_aux'] = Y_t_aux_new
                
                tasks.append(task)

        if batch_size is not None:
            task = concat_tasks_custom(tasks)
        else:
            task = tasks[batch_i]
        
        batch_loss = val_step(task)

        val_losses.append(batch_loss)
        
    return np.mean(val_losses)

def train_epoch_pickled(
    model,
    train_task_dir,
    opt,
    batch_size: int = 1,
    epoch: int = None,
    use_grad_clip: bool = False, 
    grad_clip_value: float = 0.0,
    grad_accum_steps: int = 1,
    fix_noise: float = None,
    n_subsample_targets: int = None,
    ):

    if use_grad_clip:
        from torch.nn.utils import clip_grad_norm_

    def train_step(tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        opt.zero_grad()
        task_losses = []

        for task in tasks:
            task_losses.append(model.loss_fn(task, edge_margin=(model.model.receptive_field/2), fix_noise=fix_noise, normalise=True))

        mean_batch_loss = B.mean(B.stack(*task_losses))
        (mean_batch_loss / grad_accum_steps).backward()

        if (batch_i + 1) % grad_accum_steps == 0:
            if use_grad_clip:
                clip_grad_norm_(model.model.parameters(), grad_clip_value)

            opt.step()
            opt.zero_grad()

        return mean_batch_loss.detach().cpu().numpy()

    train_task_files = [f for f in os.listdir(train_task_dir) if f.endswith('.pkl')]
    
    # limit for testing
    #train_task_files = train_task_files[:10*batch_size]

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

                if n_subsample_targets is not None:
                    X_t_new, Y_t_new, Y_t_aux_new = subsample_targets(task['X_t'], task['Y_t'], task['Y_t_aux'], n_subsample_targets, seed=epoch, replace=False)

                    task['X_t'] = X_t_new
                    task['Y_t'] = Y_t_new
                    task['Y_t_aux'] = Y_t_aux_new
                
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

def return_sample_predictions(era5_date_ds, h8_date_ds, nzra_date_ds, nzra_ds, stations_date_df, ds_aux_processed, ds_aux_coarse_processed, val_date, data_processor, epoch, model, target, mode="unmodified", padding = 200, subtitle_text="Placeholder subtitle text", title_append=""):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from deepsensor.data.loader import TaskLoader
    from convcnp_assim_nz.utils.variables.coord_names import LATITUDE, LONGITUDE

    era5_date_processed, nzra_processed, stations_date_processed = data_processor([era5_date_ds, nzra_date_ds, stations_date_df])

    temp_task_loader = TaskLoader(
        context = [stations_date_processed, era5_date_processed, ds_aux_coarse_processed], 
        target = nzra_processed,
        aux_at_targets = ds_aux_processed)
    
    # the task for inference/prediction
    task = temp_task_loader(val_date, context_sampling=["all", "all", "all"], target_sampling=["all"])

    pred = model.predict(task, X_t=nzra_ds[[LATITUDE, LONGITUDE]])

    # Number of annotated grid points
    N_ANN = 5

    if mode == "diff":
        fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    
    if title_append != "":
        fig.suptitle(f'Epoch {epoch} Predictions for {val_date} | {title_append}', fontsize=18)
    else:
        fig.suptitle(f'Epoch {epoch} Predictions for {val_date}', fontsize=18)
    
    if not mode == "diff":
        fig.text(0.5, 0.94, subtitle_text, ha='center', fontsize=12, color='gray')

    # --- DATA EXTRACTION ---
    model_target = target if mode not in ["diff"] else f"{target}_diff"
    
    truth_field = nzra_ds[model_target].sel(time=val_date).values
    pred_mean  = pred[model_target]["mean"].isel(time=0).values
    pred_std   = pred[model_target]["std"].isel(time=0).values

    if mode == "diff":
        pred_mean_adj = pred_mean + era5_date_ds[target].values
        truth_field_adj = truth_field + era5_date_ds[target].values

        pred_truth_error = truth_field_adj - pred_mean_adj

    Ny, Nx = truth_field.shape

    # add padding to the annotated points
    Ny = Ny - padding//2
    Nx = Nx - padding//2

    # Colour scale based on NZRA target image
    vmin = min(truth_field.min(), pred_mean.min())
    vmax = max(truth_field.max(), pred_mean.max())

    # use this to plot range solely from NZRA          
    #vmin = truth_field.min()
    #vmax = truth_field.max()

    # Annotated grid points
    ys = np.linspace(padding//2, Ny - 1, N_ANN, dtype=int)
    xs = np.linspace(padding//2, Nx - 1, N_ANN, dtype=int)
    annot_points = [(y, x) for y in ys for x in xs]

    # --- TOP LEFT: NZRA temperature ---
    im0 = ax[0, 0].imshow(truth_field, origin='lower', vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(f"NZRA {model_target}")
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
        
    if mode == "diff":
        # --- TOP FURTHER RIGHT: ERA5 ---
        im_extra = ax[0, 2].imshow(era5_date_ds[target].values, origin='lower')
        ax[0, 2].set_title(f"ERA5 {target}")
        fig.colorbar(im_extra, ax=ax[0, 2], shrink=0.7)
        for y, x in annot_points:
            ax[0, 2].text(x, y, f"{era5_date_ds[target].values[y,x]:.1f}", fontsize=7,
                        color='white', ha='center', va='center')

        # --- MOST BOTTOM LEFT: PRED MEAN ADJUSTED ---
        im4 = ax[2, 0].imshow(pred_mean_adj, origin='lower')
        ax[2, 0].set_title(f"ConvCNP {target} prediction (mean + ERA5)")
        fig.colorbar(im4, ax=ax[2, 0], shrink=0.7)
        for y, x in annot_points:
            ax[2, 0].text(x, y, f"{pred_mean_adj[y,x]:.1f}", fontsize=7,
                        color='white', ha='center', va='center')
            
        # --- MOST BOTTOM RIGHT: PRED ACTUAL ADJUSTED ---
        im5 = ax[2, 1].imshow(truth_field_adj, origin='lower')
        ax[2, 1].set_title(f"NZRA {target} (reconstructed)")
        fig.colorbar(im5, ax=ax[2, 1], shrink=0.7)
        for y, x in annot_points:
            ax[2, 1].text(x, y, f"{truth_field_adj[y,x]:.1f}", fontsize=7,
                        color='white', ha='center', va='center')

        # Create the colormap
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom_red_blue", ["red", "white", "blue"], N=256
        )

        # --- ERROR PLOT ---
        im6 = ax[2, 2].imshow(pred_truth_error, origin='lower', cmap=custom_cmap, norm=CenteredNorm())
        ax[2, 2].set_title(f"Prediction Error (Adjusted)")
        fig.colorbar(im6, ax=ax[2, 2], shrink=0.7)
        for y, x in annot_points:
            ax[2, 2].text(x, y, f"{pred_truth_error[y,x]:.1f}", fontsize=7,
                        color='black', ha='center', va='center')
            
        # plot with text - fills missing subplot space
        ax[1, 2].axis('off')
        ax[1, 2].text(0.5, 0.5,
                    subtitle_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12,
                    wrap=True)
        
            
    plt.tight_layout()

    return fig

import numpy as np

# for sampling from tasks 'after the fact' - i.e. sampling from a full task.

def subsample_targets(
    X_t,
    Y_t,
    Y_t_aux,
    N,
    seed=None,
    replace=False,
):
    rng = np.random.default_rng(seed)

    # Infer Nt
    Nt = X_t[0][0].shape[1]
    Ntot = Nt * Nt

    if N == 0:
        X_t_sub = [np.zeros((2, 0), dtype=X_t[0][0].dtype)]
        Y_t_sub = [np.zeros((arr.shape[0], 0), dtype=arr.dtype) for arr in Y_t]
        Y_t_aux_sub = np.zeros((Y_t_aux.shape[0], 0), dtype=Y_t_aux.dtype)
        return X_t_sub, Y_t_sub, Y_t_aux_sub

    if N > Ntot and not replace:
        raise ValueError(
            f"Cannot sample N={N} points from Nt^2={Ntot} without replacement"
        )

    # ---- sample flat grid points ----
    flat_idx = rng.choice(Ntot, size=N, replace=replace)
    ii = flat_idx // Nt
    jj = flat_idx % Nt

    # ---- X_t ----
    # original format: [(x1, x2)] where each is (1, Nt)
    x1, x2 = X_t[0]
    X_t_sub = [
        np.vstack([
            x1[:, jj],
            x2[:, ii],
        ]).astype(x1.dtype)
    ]

    # ---- Y_t ----
    Y_t_sub = []
    for arr in Y_t:
        if arr.ndim == 2:
            # (C, Nt) to (C, N)
            Y_t_sub.append(arr[:, jj])
        elif arr.ndim == 3:
            # (C, Nt, Nt) to (C, N)
            Y_t_sub.append(arr[:, ii, jj])
        else:
            raise ValueError(f"Unsupported Y_t array shape: {arr.shape}")

    # ---- Y_t_aux ----
    if Y_t_aux.ndim != 3:
        raise ValueError("Y_t_aux must have shape (C_aux, Nt, Nt)")

    Y_t_aux_sub = Y_t_aux[:, ii, jj]

    return X_t_sub, Y_t_sub, Y_t_aux_sub
