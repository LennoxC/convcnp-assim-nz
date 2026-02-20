# convcnp-assim-nz
## Update Feb 2026 | Lennox Crowe

### Project Update
The goal of the project is to make a deep-learning powered data assimilation model for accurate now-casting, for use by the energy sector, emergency responders etc. As of Feb 2026, a proof-of-concept for temperature interpolation is underway with promising results, but not completed. Once temperature interpolation is working well, we will focus on wind and solar irradiance interpolation.

Details of specific experiments are available in `experiments.md`. However, here is a broad overview of the major milestones in the project:

- Experiments 1, 2, 3: Prototyping on the university GPU servers.
- Experiment 4: 
    - Initial temperature prediction on the HPC. Some hyperparameter tuning, including gradient clipping, training on a sample of points.
    - The discovery that loss-masking around the edges of the prediction grid improved training stability.
- Experiment 5:
    - Predicting the difference of NZRA and ERA5. This resulted in collapse to the mean (but stable results).
    - Training using per-coordinate normalization.
    - Training using NZRA values for station values (training only). Switch for station temperatures at inference time. **Work in progress.**

Some notes on Experiment 5:
While NZRA and the Station temperature are highly correlated when looking at raw temperature values (see `experiment5_stations_nzra_eda.ipynb`), the per-coordinate normalized temperature residual (where the residuals are ERA5-NZRA or ERA5-Station) are not highly correlated, with an R value as low as 0.2 at some timestamps. This will be because NZRA, ERA5, and Stations have different biases. When using the normalized station residual to predict the normalized NZRA residual, the station residual may not actually be highly correlated enough to be useful. However, we want to encourage the model to output the station temperature (or close to it) at the station point, and interpolate this value to other nearby points. To encourage this, in the final experiment I used values from NZRA in place of station observations. This means the model is provided the exact target output value at a set of points, so will be able to output exactly these values at the same points, in order to see a reduction in loss. This should prevent the model from collapsing to the mean, as there are 'easy wins' to reduce negative log-liklihood. Hopefully the model will then learn how to interpolate these values to nearby points.

Obviously, NZRA data isn't available at inference time. Here, we can 'switch out' the NZRA data with station observations. These will be slightly different to the NZRA inputs, but as long as the same physical principles govern both models (i.e. NZRA is a close approximation of the real world) then the model will produce an output which may be *more accurate* than NZRA - i.e. closer to real world conditions, assuming the stations are exact measurements of temperature. The challenge becomes testing this model, as NZRA is no-longer the target. I suggest checking for close temperature correlation (unnormalized and converted back to degrees) with NZRA (i.e. R > 0.85) and also holding-out a few station observations in each task, for testing.

This experiment has not been completed - I only trained while developing on a month's worth of data. Full datasets should be generated and trained on. The results didn't show much movement in output mean, but begun to show some 'speckling' around stations after under an hour of training, which is encouraging. This method is similar to other successful applications of ConvNPs, so I think this should work.

### Notes on the environment

This project uses forks of both neuralprocesses and deepsensor. Pixi installs my fork of each (referencing a certain commit, for now) from github when you run `pixi install`. 

#### Neural Processes Modifications
There weren't any major modifications made to neuralprocesses. We noted that we wanted to try using dropout to help with normalization if we encountered overfitting. One afternoon while a model was training, I implemented dropout, but this was never actually used in an experiment. This might be useful in the future, so I still have neuralprocesses pointed at my fork. If neuralprocesses gets updated (in the main repo) then pointing this repo at the main neuralprocesses repository won't require any major code changes.

#### Deepsensor Modifications
- The major modification I made to deepsensor was loss-masking around the edge of the prediction grid. This is implemented in the `remove_edge_targets` function which is called by `loss_fn`. Just looking back at the repo now... there is an old revision of this function commented out too. This could be removed. 
- `loss_fn` has a parameter `edge_margin` to set the amount of masking around the edge of the task. This should be set to half the model receptive field.
- Some small changes to support dropout in the neuralprocesses module were added - these are just extra parameters to functions.
- A small update to the data processor in `deepsensor/data/processor.py` removes some detail from the warning message. The warning isn't important, and an exception is often thrown within the warning itself (due to a casting error I think). This was annoying and stopped notebook execution when it didn't need to stop.

In `src/convcnp-assim-nz/learning` you might notice some functions which resemble deepsensor functions. I added these due to specific requirements of this project, before I made a deepsensor fork. You would probably need to review each of these changes on a case-by-case basis to see if they should be added to my deepsensor fork, or if they can remain seperate.

`experiment5-custom-unet.ipynb` was where I did some testing for both of these modifications to the repositories. I thought I would leave this here as a template for testing if required - but note this isn't an *actual* experiment. 
