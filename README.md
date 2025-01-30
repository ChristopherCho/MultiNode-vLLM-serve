<div align="center">

# MultiNode-vLLM-serve
Run vLLM serve commands on multi-node environment

![](./images/interrogate.svg)
</div>

The repository is for running [vLLM](https://docs.vllm.ai/en/latest/) serve commands on multi-node environment via [slurm](https://slurm.schedmd.com/documentation.html).  
After all models are correctly served, you can access models with its API endpoint.  
For easy and balanced usage, you can use load balancer like [litellm](https://docs.litellm.ai/).  


## Requirements
- slurm installed and executable on all nodes
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
