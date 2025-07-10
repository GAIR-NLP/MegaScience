sudo add-apt-repository ppa:rmescandon/yq
sudo apt update
sudo apt install yq -y

# extract NNODE and NGPU from yaml
export yaml_path=./vllm_inference/task_config/distill_cot.yaml

export NNODE=$(yq eval '.N_NODES' $yaml_path)
export NGPU=$(yq eval '.NODE_GPUS' $yaml_path)
export tp=$(yq eval '.tp_size' $yaml_path)
export NUM_TASKS=$((NGPU / tp))
export save_path=$(yq eval '.save_path' $yaml_path)
export save_name=$(yq eval '.save_name' $yaml_path)

# create logging dir
export logging_dir=./logging/distill_cot
mkdir -p "${logging_dir}/${save_name}"

echo "START TIME: $(date)"

cmd="
for i in \$(seq 0 \$((${NUM_TASKS}-1))); do
    START_GPU=\$((i * \$tp)) \\
    CUDA_VISIBLE_DEVICES=\$(seq -s, \$START_GPU \$((START_GPU+\$tp-1))) \\
    python vllm_inference/distill_cot.py \\
        --config_path ${yaml_path} \\
    > ${logging_dir}/${save_name}/\${i}.log 2>&1 &
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