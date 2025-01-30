#!/bin/bash
while getopts "m:l:p:t:" opt; do
    case $opt in
        m) MODEL_PATH=$OPTARG ;;
        l) LORA_PATH=$OPTARG ;;
        t) TENSOR_PARALLEL_SIZE=$OPTARG ;;
    esac
done


# Load .env file and filter out comments & empty lines
export $(grep -v '^#' .env | xargs)


# Setup arguments
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora-modules '{\"name\": \"adapter\", \"path\": \"${LORA_PATH}\", \"base_model_name\": \"${MODEL_PATH}\"}' --enable-lora"
else
    LORA_ARG=""
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_ARG="--tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
else
    TENSOR_PARALLEL_ARG=""
fi

VLLM_ARGS="${TENSOR_PARALLEL_ARG} ${LORA_ARG}"


# Setup required ports considering tensor parallel size
PORT_POOL=($(seq ${START_PORT} $((START_PORT + NUM_GPUS_PER_NODE - 1))))
PORT_POOL_SIZE=${#PORT_POOL[@]}
if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    NUM_PORTS_TO_USE=$((PORT_POOL_SIZE / TENSOR_PARALLEL_SIZE))
else
    NUM_PORTS_TO_USE=$PORT_POOL_SIZE
fi
PORT_POOL=(${PORT_POOL[@]:0:$NUM_PORTS_TO_USE})


# Prepare variables for running tmux and vLLM serve
START_DIR=$(pwd)
NODE_ID=$(hostname)
OUTLINES_CACHE_DIR="${LOG_DIR}/outlines_cache/${NODE_ID}"


# Prepare tmux socket directory
TMUX_TMPDIR="${LOG_DIR}/tmux_sockets"
SOCKET_PATH="$TMUX_TMPDIR/$NODE_ID"
mkdir -p ${TMUX_TMPDIR}
chmod 700 ${TMUX_TMPDIR}

if [ -e "$SOCKET_PATH" ]; then
    echo "Cleaning up stale tmux socket..."
    rm -f "$SOCKET_PATH"
fi


# Warm up sessions for creating default tmux directory
tmux -S ${SOCKET_PATH} start-server
tmux -S ${SOCKET_PATH} has-session -t ${TMUX_SESSION_NAME} &> /dev/null
if [ $? -eq 0 ]; then
    tmux -S ${SOCKET_PATH} kill-session -t ${TMUX_SESSION_NAME}
fi

tmux -S ${SOCKET_PATH} new-session -s ${TMUX_SESSION_NAME} -c ${START_DIR} -d
tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.0 "cd ${START_DIR}" C-m


# Run vLLM serve on each GPU
INDEX=0
for port in ${PORT_POOL[@]}; do
    START_GPU=$((INDEX * TENSOR_PARALLEL_SIZE))
    END_GPU=$((START_GPU + TENSOR_PARALLEL_SIZE - 1))
    CUDA_VISIBLE_DEVICES=$(seq -s, $START_GPU $END_GPU)

    if [ $INDEX -ne 0 ]; then
        tmux -S ${SOCKET_PATH} split-window -t ${TMUX_SESSION_NAME}:0 -c ${START_DIR} -h
        tmux -S ${SOCKET_PATH} select-layout -t ${TMUX_SESSION_NAME}:0 tiled
    fi

    tmux -S ${SOCKET_PATH} select-pane -t ${TMUX_SESSION_NAME}:0.${INDEX}
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "cd ${START_DIR}" C-m
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "source ${CONDA_ROOT_DIR}/bin/activate" C-m
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "conda activate ${CONDA_ENV_NAME}" C-m
    # Set different OUTLINES_CACHE_DIR for each GPU to avoid error
    # Check https://github.com/vllm-project/vllm/issues/4193#issuecomment-2604001269 for more details
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "mkdir -p ${OUTLINES_CACHE_DIR}/${INDEX}" C-m
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR}/${INDEX}" C-m
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" C-m
    tmux -S ${SOCKET_PATH} send-keys -t ${TMUX_SESSION_NAME}:0.${INDEX} "vllm serve $MODEL_PATH --port $port $VLLM_ARGS" C-m

    INDEX=$((INDEX + 1))
done

echo "Node is running"

# Setup timeout
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-0}
if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
    TIMEOUT_SECONDS=$TIMEOUT_SECONDS
else
    TIMEOUT_SECONDS="infinity"
fi

sleep ${TIMEOUT_SECONDS}
