import argparse
import asyncio
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait

from dotenv import load_dotenv
from litellm import Router
from tqdm import tqdm


load_dotenv()

ROUTERS = {}

ACCESS_INFO_PATH = os.path.join(os.getenv("LOG_DIR"), "access_info")
DEFAULT_CONFIG = {
    "system_role": "system",
    "user_role": "user",
    "completion_kwargs": {
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}
MODEL_CONFIGS = defaultdict(lambda: DEFAULT_CONFIG)

MODEL_CONFIGS["mgoin/Nemotron-4-340B-Instruct-hf-FP8"] = {
    "system_role": "System",
    "user_role": "User",
    "completion_kwargs": {
        "max_tokens": 1024,
        "temperature": 0.0,
        "stop": ["<|endoftext|>", "<extra_id_1>", "\x11", "<extra_id_1>User"],
    },
}


def get_model_config(model_name):
    """Get the model config for the given model name

    Args:
        model_name (str): The name of the model

    Returns:
        dict: The model config
    """
    return MODEL_CONFIGS[model_name]


def count_instances(model_name):
    """Count the number of instances for the given model name

    Args:
        model_name (str): The name of the model

    Returns:
        int: The number of instances
    """
    router = _get_router(model_name)
    return len(router.model_list)


def build_router(model_name):
    """Build the router for the given model name

    Args:
        model_name (str): The name of the model

    Returns:
        bool: True if the router is built successfully, False otherwise
    """
    print(f"Establishing router for {model_name}")

    model_access_info_path = os.path.join(ACCESS_INFO_PATH, f"{model_name}.json")
    with open(model_access_info_path, "r") as f:
        access_info = json.load(f)

    ROUTERS[model_name] = Router(model_list=access_info)

    try:
        model_config = get_model_config(model_name)
        system_prompt = "You are a helpful assistant."
        user_prompt = f"Print out the text: {model_name} ready!"
        messages = [
            {"role": model_config["system_role"], "content": system_prompt},
            {"role": model_config["user_role"], "content": user_prompt},
        ]
        response = asyncio.run(request_model(model_name, messages, model_config["completion_kwargs"]))
        print(f"Router for {model_name} established successfully - {response}")
    except Exception:
        return False

    return True


def _get_router(model_name):
    """Get the router for the given model name

    Args:
        model_name (str): The name of the model

    Raises:
        ValueError: If the router for the given model name is not found

    Returns:
        Router: The router for the given model name
    """
    if model_name in ROUTERS:
        router = ROUTERS[model_name]
    else:
        raise ValueError(f"Router for {model_name} not found")
    return router


def build_messages(system_role, user_role, system_prompt, user_prompt):
    """Build the messages for the given system role, user role, system prompt, and user prompt

    Args:
        system_role (str): The system role (e.g. "system")
        user_role (str): The user role (e.g. "user")
        system_prompt (str): The system prompt
        user_prompt (str): The user prompt

    Returns:
        list: The messages to be fed into the model
    """
    return [{"role": system_role, "content": system_prompt}, {"role": user_role, "content": user_prompt}]


async def request_model(model_name, messages, additional_args):
    """Request the model for the given model name, messages, and additional arguments

    Args:
        model_name (str): The name of the model
        messages (list): The messages to be fed into the model
        additional_args (dict): The additional arguments for completion

    Returns:
        str: The response from the model
    """
    router = _get_router(model_name)
    response = await router.acompletion(model=model_name, messages=messages, **additional_args)
    return response.choices[0].message.content.strip()


async def process_sample(model_name, sample):
    """Process the sample for the given model name and sample

    Args:
        model_name (str): The name of the model
        sample (dict): The sample to be processed

    Returns:
        dict: The result of the processing
    """
    system_prompt = "You are a helpful assistant."
    user_prompt = sample["user_prompt"]

    model_config = get_model_config(model_name)
    messages = build_messages(model_config["system_role"], model_config["user_role"], system_prompt, user_prompt)
    response = await request_model(model_name, messages, model_config["completion_kwargs"])

    return {"response": response, **sample}


async def process_model_requests(model_name, samples, output_file_path, model_idx, max_concurrent_tasks=100):
    """Process the model requests for each sample and save the result

    Args:
        model_name (str): The name of the model
        samples (list): The samples to be processed
        output_file_path (str): The path to the output file
        model_idx (int): The index of the model
        max_concurrent_tasks (int, optional): The maximum number of concurrent tasks. Defaults to 100.

    Returns:
        None
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def process_model_requests_with_limit(sample):
        """Process the sample for the given model name and sample with a limit on the number of concurrent tasks

        Args:
            sample (dict): The sample to be processed

        Returns:
            dict: The result of the processing
        """
        async with semaphore:
            try:
                return await process_sample(model_name, sample)
            except Exception as e:
                return {"error": str(e)}

    tqdm_bar = tqdm(desc=f"Processing {model_name}", position=model_idx)
    jobs = [process_model_requests_with_limit(sample) for sample in samples]

    with open(output_file_path, "w") as f:
        for coro in asyncio.as_completed(jobs):
            result = await coro
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            tqdm_bar.update(1)

    tqdm_bar.close()


def run_async(model_name, samples, output_file_path, model_idx, max_concurrent_tasks):
    """Async wrapper for running process_model_requests with multiprocessing

    Args:
        model_name (str): The name of the model
        samples (list): The samples to be processed
        output_file_path (str): The path to the output file
        model_idx (int): The index of the model
        max_concurrent_tasks (int): The maximum number of concurrent tasks

    Returns:
        str: The name of the model
    """
    asyncio.run(process_model_requests(model_name, samples, output_file_path, model_idx, max_concurrent_tasks))

    return model_name


def main(args):
    # Build the router for each model
    for model_name in args.model_names:
        build_router(model_name)

    # Prepare dummy input samples
    samples = [{"user_prompt": "Hello, how are you?"} for _ in range(100)]

    # Run the async process for each model
    futures = []
    with ProcessPoolExecutor(max_workers=len(args.model_names)) as executor:
        for model_idx, model_name in enumerate(args.model_names):
            output_file_path = os.path.join(args.output_dir, f"{model_name}.jsonl")
            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            num_instances = count_instances(model_name)
            max_concurrent_tasks = args.concurrent_tasks_per_instance * num_instances

            futures.append(
                executor.submit(run_async, model_name, samples, output_file_path, model_idx, max_concurrent_tasks)
            )

    # Wait for all the tasks to complete
    done, _ = wait(futures)
    for future in done:
        try:
            model_name = future.result()  # This retrieves results or raises exceptions
            print(f"Model {model_name} completed")
        except Exception as e:
            print(f"Error during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-names", nargs="+", required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("--concurrent-tasks-per-instance", type=int, default=10)
    args = parser.parse_args()

    main(args)
