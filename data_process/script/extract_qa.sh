# sudo add-apt-repository ppa:rmescandon/yq
# sudo apt update
# sudo apt install yq -y

chmod +x /inspire/hdd/global_user/liupengfei-24025/global/rzfan/yq_linux_amd64
cp -r /inspire/hdd/global_user/liupengfei-24025/global/rzfan/yq_linux_amd64 /usr/local/bin/yq
yq --version

# setup env
export PATH="/inspire/hdd/global_user/liupengfei-24025/global/rzfan/miniconda3/bin:$PATH"
source activate factory
cd /inspire/hdd/global_user/liupengfei-24025/global/rzfan/vllm_inference

# extract NNODE and NGPU from yaml
export yaml_path=/inspire/hdd/global_user/liupengfei-24025/global/rzfan/vllm_inference/task_config/extract_reference_answer/biology_extract_reference_answer.yaml

export NNODE=$(yq eval '.N_NODES' $yaml_path)
export NGPU=$(yq eval '.NODE_GPUS' $yaml_path)
export tp=$(yq eval '.tp_size' $yaml_path)
export NUM_TASKS=$((NGPU / tp))
export save_name=$(yq eval '.save_name' $yaml_path)

# create logging dir
export logging_dir=./logging/extract_reference_answer
mkdir -p "${logging_dir}/${save_name}"

echo "START TIME: $(date)"

cmd="
for i in \$(seq 0 \$((${NUM_TASKS}-1))); do
    FORCE_TORCHRUN=1 \\
    NNODES=${PET_NNODES} \\
    NODE_RANK=${PET_NODE_RANK} \\
    MASTER_ADDR=${MASTER_ADDR} \\
    MASTER_PORT=${MASTER_PORT} \\
    START_GPU=\$((i * \$tp)) \\
    CUDA_VISIBLE_DEVICES=\$(seq -s, \$START_GPU \$((START_GPU+\$tp-1))) \\
    /inspire/hdd/global_user/liupengfei-24025/global/rzfan/miniconda3/envs/factory/bin/python extract_reference_answer.py \\
        --config_path ${yaml_path} \\
        > ${logging_dir}/${save_name}/\${SLURM_NODEID}_\${i}.log 2>&1 &
done
wait
"

echo "Executing command:"
echo "$cmd"

bash -c "$cmd"

# calculate the total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Total time: $total_time seconds"