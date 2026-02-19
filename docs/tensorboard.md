## Viewing experiments in Tensorboard

All the charts, loss curves etc generated in each model run are stored in tensorboard. This tensorboard directory on the HPC is `/home/crowelenn/dev/convcnp-assim-nz/.tensorboard/experiment4/runs`.

Tensorboard can be started by running `scripts/utils/start_tensorboard.sh`. This starts a tensorboard on the specified port (default is 6006) using nohup, meaning that the tensorboard session will persist even if you shut the terminal, or log on/off the HPC. Note that I used VSCode's default port forwarding to be able to view tensorboard in my own browser, but you may need to manually set up port forwarding if not using vscode.

To start tensorboard in the directory where I did my experiments, run: `./scripts/utils/start_tensorboard.sh runs 6006 /home/crowelenn/dev/convcnp-assim-nz/.tensorboard/experiment4` from the repository root - i.e. `/home/$USER/dev/convcnp-assim-nz`.

To shut down tensorboard, run `ps aux | grep 6006` (or specify the correct port if not using 6006) to find the PID. Then run `kill <PID>` where `<PID>` is the process identifier returned from the previous command. Alternatively, when you start running tensorboard using the provided script, the PID will be printed to the terminal. You can remember this PID to shut the process down later.