from deepsensor.data.utils import construct_x1x2_ds
from convcnp_assim_nz.utils.variables.coord_names import *

def build_cropped_auxiliary(ds_aux, ds_aux_crop, ds_aux_processed, ds_aux_crop_processed):
    """
    In the case when you have trained on a wider area and now want to apply the model on a cropped area,
    you need to build new positional encodings for the cropped area while keeping the original ones for
    the wider area.
    This function builds such positional encodings. 
    - ds_aux: original auxiliary dataset (wider area)
    - ds_aux_crop: cropped auxiliary dataset (smaller area)
    - ds_aux_processed: processed original auxiliary dataset with positional encodings
    - ds_aux_crop_processed: processed cropped auxiliary dataset with positional encodings
    'processed' referes to the deepsensor data processor which normalises the coordinates
    """
    
    x1x2_ds = construct_x1x2_ds(ds_aux_processed)
    x1x2_ds_crop = construct_x1x2_ds(ds_aux_crop_processed)

    ds_aux_processed['x1_arr'] = x1x2_ds['x1_arr']
    ds_aux_processed['x2_arr'] = x1x2_ds['x2_arr']

    # if the new index ds_aux[LATITUDE] or ds_aux[LONGITUDE] is longer than the ds_aux_processed x1 or x2 coords by no more than 1, 
    # we can crop the new latitude/longitude to match the processed coords
    # this sometimes happens due to rounding errors in the data processor when mapping the coordinates

    x1_diff = len(ds_aux[LATITUDE]) - len(ds_aux_processed['x1'])
    x2_diff = len(ds_aux[LONGITUDE]) - len(ds_aux_processed['x2'])

    if x1_diff > 1 or x2_diff > 1:
        raise ValueError("The difference in length between the original and processed coordinates is too large.")
    
    if x1_diff == 1:
        ds_aux_processed = ds_aux_processed.assign_coords(x1=ds_aux[LATITUDE].values, x2=ds_aux[LONGITUDE].values[:-1])
    elif x2_diff == 1:
        ds_aux_processed = ds_aux_processed.assign_coords(x1=ds_aux[LATITUDE].values[:-1], x2=ds_aux[LONGITUDE].values)
    else:
        ds_aux_processed = ds_aux_processed.assign_coords(x1=ds_aux[LATITUDE].values, x2=ds_aux[LONGITUDE].values)
    
    ds_aux_processed_latidx = ds_aux_processed.sel(
        x1=slice(ds_aux_crop[LATITUDE].min(), ds_aux_crop[LATITUDE].max()),
        x2=slice(ds_aux_crop[LONGITUDE].min(), ds_aux_crop[LONGITUDE].max())
    )

    ds_aux_new = ds_aux_processed_latidx
    ds_aux_new = ds_aux_new.assign_coords(x1=ds_aux_crop_processed['x1'], x2=ds_aux_crop_processed['x2'])
    ds_aux_new['x1_arr_l1'] = ds_aux_processed_latidx.assign_coords(x1=ds_aux_new['x1'].values, x2=ds_aux_new['x2'].values)['x1_arr']
    ds_aux_new['x2_arr_l1'] = ds_aux_processed_latidx.assign_coords(x1=ds_aux_new['x1'].values, x2=ds_aux_new['x2'].values)['x2_arr']
    ds_aux_new['x1_arr_l2'] = x1x2_ds_crop['x1_arr']
    ds_aux_new['x2_arr_l2'] = x1x2_ds_crop['x2_arr']
    ds_aux_new = ds_aux_new.drop_vars(['x1_arr', 'x2_arr'])

    return ds_aux_new
