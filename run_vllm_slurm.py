import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv


load_dotenv()

LOG_DIR = os.getenv("LOG_DIR")
CWD = os.path.dirname(os.path.realpath(__file__))

TEMPLATE = """#!/bin/bash
#SBATCH --partition={slurm_partition}
#SBATCH -o {log_path}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=32G

export MASTER_ADDR=$(hostname)

echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master addr: $MASTER_ADDR"

srun -l sh -c 'bash main.sh -m {model_path} {additional_args}'
"""


def get_node_names(node_setup_string: str):
    """Get node names from the node setup string.

    Args:
        node_setup_string (str): The node setup string. Starts with "Nodes: ".

    Returns:
        List[str]: The list of node names.
    """
    nodes = node_setup_string.replace("Nodes: ", "").strip()

    # Find the list notation in the string
    list_pattern = r"\[[^\]]+\]"
    list_notation = re.findall(list_pattern, nodes)
    assert len(list_notation) <= 1, f"Invalid nodes format: {nodes}"

    # If there is no list notation, return the nodes as a single element list
    if len(list_notation) == 0:
        return [nodes]

    # If there is a list notation, extract the nodes from the list
    matches = re.fullmatch(rf"(.*)({list_pattern})(.*)", nodes)
    prefix, list_content, suffix = matches.groups()

    # For each node in the list, extract the node id
    nodes_list = list_content[1:-1].split(",")
    node_ids = []
    for node_id in nodes_list:
        if "-" in node_id:
            start, end = node_id.split("-")
            node_ids.extend(range(int(start), int(end) + 1))
        else:
            node_ids.append(node_id)

    # Return the nodes with the prefix and suffix
    return [f"{prefix}{node_id}{suffix}" for node_id in node_ids]


def main(args: argparse.Namespace):
    # Check if the timeout is set to INFINITY
    timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", -1))
    if timeout_seconds == -1:
        print("+=====================================[WARNING]=====================================+")
        print("|                            Timeout is set to INFINITY.                            |")
        print("| This means the vLLM serve will run indefinitely even if the model is not working. |")
        print("|      It is recommended to set a timeout for better GPU resource utilization.      |")
        print("+===================================================================================+")
        input("Press Enter to continue... (Ctrl+C to cancel)")
    else:
        # Calculate the ETA of the job
        now = datetime.now().astimezone()
        eta = now + timedelta(seconds=timeout_seconds)
        eta_str = eta.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"Timeout: {timeout_seconds} seconds (ETA: {eta_str})")

    # Setup variables for the job
    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{args.job_name}-{execution_id}"
    runfile_path = os.path.join(LOG_DIR, "scripts", f"{args.job_name}_{execution_id}.slurm")
    log_path = os.path.join(LOG_DIR, "logs", f"{args.job_name}_{execution_id}.log")

    # Write the runfile
    with open(runfile_path, "w") as f:
        lora_arg = f"-l {args.lora_path}" if args.lora_path else ""
        tensor_parallel_arg = f"-t {args.tensor_parallel_size}" if args.tensor_parallel_size else ""
        additional_args = f"{lora_arg} {tensor_parallel_arg}"

        data = TEMPLATE.format(
            slurm_partition=os.getenv("SLURM_PARTITION"),
            log_path=log_path,
            job_name=job_name,
            nodes=args.nodes,
            model_path=args.model_path,
            additional_args=additional_args,
        )
        f.write(data)

    # Run the job with sbatch
    print(f"Running job on {CWD}")
    subprocess.run(["sbatch", runfile_path], cwd=CWD)

    # Check the job status
    time.sleep(2)  # Wait for the job to start
    result = subprocess.run(["squeue", "--me"], capture_output=True, text=True)
    print(result.stdout)

    # Wait for the job to be started and the log file to be created
    while True:
        time.sleep(1)
        if not os.path.exists(log_path):
            continue

        with open(log_path, "r") as f:
            nodes = f.readlines()[0]

        if not nodes.startswith("Nodes: "):
            continue

        nodes_list = get_node_names(nodes)
        if len(nodes_list) != args.nodes:
            continue

        break

    # Get the access info and save it
    access_info_path = os.path.join(LOG_DIR, "access_info", f"{args.model_path}.json")
    if not os.path.exists(os.path.dirname(access_info_path)):
        os.makedirs(os.path.dirname(access_info_path), exist_ok=True)

    access_info = [
        {
            "model_name": args.model_path,
            "litellm_params": {
                "model": f"hosted_vllm/{args.model_path}",
                "api_key": "token-123",  # Dummy API key
                "api_base": f"http://{node_id}:{int(os.getenv('START_PORT')) + i}/v1",
            },
        }
        for node_id in nodes_list
        for i in range(8 // int(args.tensor_parallel_size))
    ]

    with open(access_info_path, "w") as f:
        json.dump(access_info, f, ensure_ascii=False, indent=2)

    # Check the accessability of the model
    if args.check_access:
        print("Checking accessability of the model...")
        import litellm
        from litellm import Router

        litellm._logging._disable_debugging()  # Suppress logs

        accessability = [False for _ in range(len(access_info))]

        while True:
            for idx, instance_info in enumerate(access_info):
                router = Router(model_list=[instance_info])
                try:
                    response = router.completion(
                        model=f"hosted_vllm/{instance_info['model_name']}",
                        messages=[{"role": "user", "content": 'Tell me "Hello, world!" without any additional text.'}],
                        **{"max_tokens": 50, "temperature": 0.0},
                    )
                    response_text = response.choices[0].message.content
                    print(
                        f"Accessable: {instance_info['model_name']} ({instance_info['litellm_params']['api_base']}) - {response_text}"
                    )
                    accessability[idx] = True
                except Exception:
                    continue

            if all(accessability):
                break

            total = len(access_info)
            accessable = sum(accessability)
            inaccessible = total - accessable
            print(f"({args.model_path}) {inaccessible}/{total} models are not accessible yet. Waiting for 1 minute...")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", "-j", type=str, required=True, help="Name of the job")
    parser.add_argument("--nodes", "-n", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Name of the model. Should be a Hugging Face model name. (e.g. upstage/solar-pro-preview-instruct)")
    parser.add_argument("--tensor-parallel-size", "-t", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--lora-path", type=str, help="Path to the lora model")
    parser.add_argument("--check-access", action="store_true", help="Validate accessability of the model")
    args = parser.parse_args()
    main(args)
