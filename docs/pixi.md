# Details of the Pixi Environment

While developing the ConvCNP pipeline, notebooks have been running on a kernel hosted by `jupyter.slv.hpc.niwa.co.nz`, on a node called `hub-worker01.slv`. Development is done in VSCode. To set up VSCode development, a token was configured in jupyter hub (file > Hub Control Panel > Token > *fill out form and request new API token* > copy the token and store it). To connect to this kernel with VSCode: 

- start your jupyter server
- open a jupyter notebook in VSCode and click `select kernel` (or equivilant) in the top right hand corner
- Choose 'select another kernel' then 'Existing Jupyter Server'
- input the url: https://jupyter.slv.hpc.niwa.co.nz/user/<your-hub-user-name>/?token=<your-token>

You should be able to run the notebook against the jupyter compute, and still have access to your login node terminal in VSCode, which is very convenient.

### Submitting jobs to the HPC

The kernel is a property of the notebook. I.e. if you select the jupyter hub kernel and do your notebook development against that, when you submit the job to the HPC, the kernel will not be changed, and the code will execute on the Jupyter hub node (not the new HPC node). This becomes obvious when you submit a job to GPU compute, but then you fail to connect to a GPU (this is how I noticed this was an issue). This is also a waste of HPC resources - effectively using the HPC as a master node which does no compute!

First, to handle differing pytorch dependencies between CPU compute and GPU compute, I created multiple pixi environments: cpu (default) and gpu. To see how the multiple environments are configured, see the project `pyptoject.toml` file. When using pixi, the -e flag allows you to specify the environment. E.g. `pixi run -e gpu python -m ...` will run a python module in the GPU environment. To install the GPU environment, I had to do this on the login node (with internet access). You can trick the login node to thinking it has a GPU by running `export CONDA_OVERRIDE_CUDA=12.8`. Then `pixi install -e gpu` installs the GPU environment. As the filesystem is shared, this environment will be accessible from GPU nodes. Next you need to create a kernel which has access to the GPU dependencies. run: `pixi run -e gpu python -m ipykernel install --user --name convcnp-gpu`. This kernel will have access to the pythorch GPU dependencies and can be used by HPC GPU nodes. 

To submit a notebook job and specify which kernel to use, see the demo below. Note the line: `--ExecutePreprocessor.kernel_name=convcnp-gpu`.

```
pixi run -e gpu python -m nbconvert \
  --to notebook \
  --execute "./notebooks/experiment4/experiment4_nzra_target.ipynb" \
  --output "experiment4_nzra_target_executed.ipynb" \
  --output-dir "./notebooks/experiment4/executed" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=convcnp-gpu
```

This will ensure that the notebook uses the correct kernel, with the correct dependencies, and the correct hardware.