<div align="center">

# MultiNode-vLLM-serve
Run vLLM serve commands on multi-node environment

![](./images/interrogate.svg)
</div>

The repository is for running [vLLM](https://docs.vllm.ai/en/latest/) serve commands on multi-node environment via [slurm](https://slurm.schedmd.com/documentation.html).  
After all models are correctly served, you can access models with its API endpoint.  
For easy and balanced usage, you can use load balancer like [litellm](https://docs.litellm.ai/).  


## Requirements
- tmux and slurm installed on all nodes
- shared storage mounted on all nodes by the same path


## Setup
1. Clone the repository on the shared storage
1. Navigate to the repository directory
1. Copy `.env_example` to `.env` and set the environment variables
1. Create a conda environment with the name specified in `.env`
    ```bash
    conda create -n vllm-serve python=3.10
    conda activate vllm-serve
    ```
1. Install the dependencies
    ```bash
    make install
    ```

### Environment Variables
```
LOG_DIR=/path/to/log/dir

SLURM_PARTITION=default
START_PORT=40020
NUM_GPUS_PER_NODE=8
TMUX_SESSION_NAME=vllm-serve

CONDA_ROOT_DIR=/path/to/conda/root/dir
CONDA_ENV_NAME=vllm-serve

TIMEOUT_SECONDS=86400
```
- `LOG_DIR`: Directory to store the log files. Should be a shared storage.
- `SLURM_PARTITION`: Slurm partition name
- `START_PORT`: Starting port number for the vLLM serve. vLLM serve will be served on `START_PORT` + `INDEX`.
- `NUM_GPUS_PER_NODE`: Number of GPUs per node
- `TMUX_SESSION_NAME`: Name of the tmux session
- `CONDA_ROOT_DIR`: Root directory of the conda environment. Should be a shared storage.
- `CONDA_ENV_NAME`: Name of the conda environment
- `TIMEOUT_SECONDS`: Timeout seconds for the vLLM serve. If set to `-1`, the vLLM serve will run indefinitely even if the model is not working.


## Usage
```
usage: run_vllm_slurm.py [-h] --job-name JOB_NAME [--nodes NODES] --model-path MODEL_PATH [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--lora-path LORA_PATH]
                         [--check-access]

options:
  -h, --help            show this help message and exit
  --job-name JOB_NAME, -j JOB_NAME
                        Name of the job
  --nodes NODES, -n NODES
                        Number of nodes to use
  --model-path MODEL_PATH, -m MODEL_PATH
                        Name of the model. Should be a Hugging Face model name. (e.g. upstage/solar-pro-preview-instruct)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -t TENSOR_PARALLEL_SIZE
                        Tensor parallel size
  --lora-path LORA_PATH
                        Path to the lora model
  --check-access        Validate accessability of the model
```

### How it works
1. After running `run_vllm_slurm.py`, the script will create a slurm job and run the vLLM serve on each Node.
1. The slurm job will create a tmux session on each node that runs the vLLM serve on each pane.
    - The number of models served on a single node is determined by `NUM_GPUS_PER_NODE` / `TENSOR_PARALLEL_SIZE`.  
    - For each node, the first `NUM_GPUS_PER_NODE` GPUs (or less if `NUM_GPUS_PER_NODE` is not divisible by `TENSOR_PARALLEL_SIZE`) will be used.
    - Each model will be served on a different port (starting from `START_PORT`).
1. When the `TIMEOUT_SECONDS` is reached, the slurm job will be terminated.


## Examples
### Single-node single-model serve
```bash
python run_vllm_slurm.py -j vllm-serve-solar -n 1 -m upstage/solar-pro-preview-instruct --check-access
```

### Multi-node (4 nodes) single-model serve
```bash
python run_vllm_slurm.py -j vllm-serve-solar -n 4 -m upstage/solar-pro-preview-instruct --check-access
```

### Multi-node (8 nodes) multi-model (4 models) serve
```bash
python run_vllm_slurm.py -j vllm-serve-solar -n 2 -m upstage/solar-pro-preview-instruct --check-access \
& python run_vllm_slurm.py -j vllm-serve-gemma -n 2 -m google/gemma-2-27b-it -t 2 --check-access \
& python run_vllm_slurm.py -j vllm-serve-phi4 -n 2 -m microsoft/phi-4 --check-access \
& python run_vllm_slurm.py -j vllm-serve-mistral -n 2 -m mistralai/Mistral-Nemo-Instruct-2407 --check-access
```
(Note: Make sure to set a different job_name for each model to avoid log file name conflicts.)


## Test
Check out the [`test.py`](./test.py) for getting the response from each model.
```bash
# Multi-node multi-model test
python test.py \
    -m upstage/solar-pro-preview-instruct \
       google/gemma-2-27b-it \
       microsoft/phi-4 \
       mistralai/Mistral-Nemo-Instruct-2407 \
    -o output_dir
```
