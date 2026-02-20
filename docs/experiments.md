# Experiments

Please ensure that you have tensorboard running, and pointing at the `/home/crowelenn/dev/convcnp-assim-nz/.tensorboard/experiment4/runs` directory. The experiments are named by what is listed in tensorboard.

As outlined in both *update-feb-2026* and *notebooks.md* there were a few major milestones in this project, and multiple training runs within each milestone where different hyperparameters were tested. This document groups the tensorboard runs together into before/after the different milestones. Note that this isn't a complete list of every run which was ever logged to tensorboard - there were hundreds more which ended in an exception being thrown, or something breaking. These were removed from tensorboard. In this document, I don't discuss the results in depth (I usually provide a brief summary) - what I am putting more focus on is the motivation for making changes, and what changed between experiments.

The diagnostics in tensorboard are similar for each experiment. I will refer to the following:
- `Hyperparameters/Learning_Rate`: The learning rate is plotted at each step to the tensorboard time series panel. This is useful when using learning rate scheduling.
- `Loss/Train` and `Loss/Validation`: The negative-log-likelihood on the entire test set and validation set are plotted to these time series panels. Often (but not always) the validation loss is only logged every fifth epoch. Calculating the validation loss takes a long time on large datasets so isn't a priority.
- `Predictions/Train` and `Predictions/Validation`: Some sample prediction images from the ConvNP. The train image makes a prediction on training data, and the validation image makes a prediction on validation data. The format of this image can change between experiments, but all include the target (often NZRA), the model prediction grid, and the model variance grid.
- `System/CUDA_Memory_MB`: Used to plot how much VRAM is being used. The A100's we train on have ~80gb VRAM.
- `Tasks/Example`: An example task showing the context sets and target set. This changes very frequently between experiments.
- **HPARAMS**: The tensorboard tab to log model hyperparameters. This tracks a set of hyperparameters between experiments. 

## Temperature prediction: Stations to NZRA

### 48h_large_unet
See the task plot - most variables available to us were included in this first training run. This is essentially the 'baseline' model: use himawari, stations, ERA5, sun angle/culmination, and auxiliary information. The results were interesting - it was actually the lowest loss achieved in the whole project, but the predictions appear very unstable - there are large 'blobs' over the ocean which do not exist in the training data. Variance gets up to 7 degrees which is high - but technically correct (as in the target temperature probably lies within this uncertainty). The model is very uncertain about the predictions. This model is not good enough to be used in production, so further experiments were conducted to improve it. This model is somewhat of an outlier - similar experiments (like 48h_small_unet) were not able to achieve loss anywhere near as good, despite only tweaking hyperparameters.

I should note that there were a few 'debug' experiments conducted before this one while setting up the codebase - but this was the first experiment worth keeping. The experiments below are listed in the order of which they were conducted. Gradient clipping was identified as a requirement in one of these earlier experiments due to the unstable model fitting process.

### 48h_small_unet

48h_large_unet used a unet of shape (32, 64, 128, 256, 512). Note that this doesn't include the layer from the number of input channels to the first unet layer of specified channels. e.g. 22 -> 32. This Unet will also 'shrink' back down again. 48h_small_unet used a unet of shape (64, 64, 64, 64, 64), and got far worse results. This encouraged me to use a unet architecture similar to 48h_large_unet for most of my experiments. This is logged in tensorboard HPARAMS.

### fixed_sampling_per_date

For each timestamp, I sampled the same points each time. This is related to the `n_subsample_targets` hyperparameter which controls how many target points are sampled from when performing model training. The results were no better than 48h_small_unet so this approach was abandoned. The predictions are still very unstable.

### full_target_fixed_noise

I attempted to not use target sampling, and to train a model on the full target grid. This however led to a LinAlgError, due to a matrix having negative eigenvalues. I'm pretty sure the matrix in question was the variance matrix, so negative values don't make sense. More research into this issue may be needed. A solution to this proposed by neuralprocesses was to set the fix_noise parameter to add a small noise value onto the matrix diagonal, rather than noise being learnable. This should stabilize training. Loss spiked to ~5000 and the outputs were nonsense

### high_density_fixed_noise_for_25

I tried to set the fix_noise parameter to 0 after 25 epochs, hoping training had stabilized by then. This did not work, the LinAlg error was thrown after 25 epochs. I believe a lower fix_noise value was used and this reduced the loss (but the loss was still high). After this, noise fixing was not used in any other experiment. Training on the full target set became possible after introducing edge masking, however edge masking did not fully solve the problem. It is still possible to encounter the LinAlgError if you make training unstable in some other way, for example, by setting the learning rate too high. I encountered this in later experiments.

### unstable_underfit

We got additional data from 2013 to 2018, and started using this data from this point onwards. I attempted to recreate the success of 48h_large_unet with this extra data, but instead got really unstable training, and the model did not converge to anything. I believe I still only used a subset of the extra data, as n_samples in the hyperparameters is 6986, but is 15622 for the next experiment.

### unstable_underfit_2

I trained for less epochs, decreasing the learning rate quicker to try to encourage some stability. All the data available was used to train the model. The results were similar to unstable_underfit. This is the experiment which made me notice the instability around the edges.

### large_batch_learnt_stations

I reduced the task complexity massively to try to encourage the model to fit. Only the station temperature + era5 temperature along with other auxiliary variables were used. The model did not converge to anything useful (was still very unstable).

## Masking the Loss around the edge of the prediction grid
All experiments up to this point were affected by instability around the edge of the prediction grid.

### edge_loss_masking_with_gradclip
Edge loss masking caused the model to produce much more sensible predictions, but the training and validation loss stayed quite high and didn't converge. I realized that I had left on gradient clipping, and this wasn't needed now that the model training was more stable.

### edge_loss_masking_1
Same as edge_loss_masking_with_gradclip except without gradient clipping. The training and validation losses reduced further, and the output predictions in both train and validation were both the most sensible predictions that were achieved througout the 'temperature prediction' part of the experiment. I noticed that the model had largely just converged to the annual mean temperature per coordinate, and wasn't exploring how the station observations, sun angle/culmination etc affect temperature. I had one last attempt (next experiment) before pivoting to prediction the difference of NZRA-ERA5, hoping that this would avoid mean convergence.

### sampling_10000
I wondered if the model had underfit (converged to the mean) because gradient descent was limited by the sample of only 1,000 points from the target. I increased target sampling to 10,000 but the model didn't appear to be improving anymore, and was not outperforming edge_loss_masking_1 after 20 epochs.

## Predicting the difference of ERA5-NZRA + Start of Experiment 5

I shifted to start predicting the difference of NZRA-ERA5, in hopes that the model would not be able to converge to a mean at each coordinate, by predicting this residual. To summarise - the model converged to the bias of ERA5 (when compared to NZRA). It turns out ERA5 is a biased predictor of NZRA, and the model just converged to this bias. For example, ERA5 always overestimates the temperature of the southern alps. The model just predicted it was always colder in the southern alps, and largely ignored the sensors.

### test_diff
This was a small test on a subset of the dataset, just to verify the new prediction pipeline was working. The initial results looked promising so I left this experiment in tensorboard.

### predict_diff_1
The first time predicting the difference between ERA5 and NZRA after training on the full dataset. This was very stable, but quickly converged to the mean bias of ERA5, and plateaued, where I cancelled the experiment.

### predict_diff_smaller_unet
Just to see if the model was underfitting due to the complexity of the model, I reduced the unet size. The training and validation loss curves were much more stable than they were previously (i.e. less fluctuation in loss while training) but the model did not converge any further than the previous model.

## Normalizing by Coordinate on ERA5-NZRA
The motivation and implementation for this normalization is detailed extensively in `experiment5_stations_target.ipynb`. This also includes an explanation for the next stage of the experiment: Using NZRA as Stations Observations. I would reccomend reading this! All the experiments from here on use the custom_normalizer module to normalize the residuals of ERA5 to stations and NZRA datasets by coordinate. Other datasets (sun, elevation, ERA5 fields which aren't temperature) use the default deepsensor global normalization.

### per-coord-norm-reduced-task
A small task which only included normalized station observations, some auxiliary measurements, and the target set. This was the first time training on data which was normalized by coordinate, and after learning my lesson that starting 'simpler is better', I made sure to start with the minimum viable implementation - i.e. with a small task without too many channels. While this should stop the model from collapsing to the mean (because we have normalized to the mean) the model just refused to predict anything other than 0 at all points! So it still 'collapsed to mean', just there was no collapse because it started at the mean of zero.

### per-coord-norm-with-era5
I tried adding in some extra channels into the task which might be useful for the prediction task, to try to encourage some gradients to be generated. This didn't make much difference. The model still outputted a mean of roughly zero.

### per-coord-norm-reduced-task-small-unet
I tried reducing the unet size in an attempt to 'overfit' rather than the chronic underfitting which we were experiencing, but this didn't make a difference.

## Using NZRA as Station Observations
I did some EDA and found that there isn't a strong relationship between normalized station residuals and normalized ERA5 residuals, so the model might be ignoring station observations because they aren't useful enough to predict the NZRA target. I suggest training on samples from NZRA in place of the station observations, and then 'switching out' the NZRA samples for station observations after training. This presents challenges for testing. You could hold out some station observations and evaluate accuracy at these points, or just check for general close correlation (but not exact correlation) with NZRA.

When training, you don't need to limit yourself to station locations. You could train the model on points sampled from anywhere over land, and then at inference time just feed in station observations from station locations. A hybrid approach could be used where the model is trained on a locations across the country, and then fine-tuned on station locations.

I ran out of time to test this on the full dataset. I think this is going to be really promising. Other studies successfully use ConvNPs to do interpolation tasks just like this one. I did two experiments on Jan 2013 for testing, but didn't have time to generate the full dataset, and then train the model.

Note, only NZRA points over land were used for training (for the first time).

### test-nzra-as-stations
A very small task of NZRA samples, sun_angle, elevation, x1, x2 -> target NZRA was used. I only trained for under two hours on one months worth of data. I am encouraged by the speckling of the variance + mean (if you look closely) around stations. I think the model is just starting to use station observations.

### test-nzra-as-stations-steep-lr
I used a higher learning rate which converged faster, and the model kept improving across all epochs. This shows the model still has plenty of learning to do.

That was all the experiments I had time for!