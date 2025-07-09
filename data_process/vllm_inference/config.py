import os
import re
from dataclasses import dataclass
from typing import Optional, TypeVar

import yaml

DataclassT = TypeVar("DataclassT")


class GentaskConfig:
    "Model Loading Configs"
    model_path: Optional[str] = None
    tp_size: Optional[int] = 1
    gpu_memory_utilization: Optional[float] = 0.9

    "Sampling Configs"
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2000
    n_sample: Optional[int] = 1

    "Distribution Configs"
    N_NODES: Optional[int] = 1  # number of nodes
    NODE_GPUS: Optional[int] = 8  # number of gpus per node
    NODE_RANK: Optional[int] = 0  # rank of the node
    GLOBAL_RANK: Optional[int] = 0  # rank of the distributed task
    TOTAL_SPLIT: Optional[int] = 1  # total number of splits

    "Data Configs"
    data_path: Optional[str] = None
    prompt_path: Optional[str] = None
    document_path: Optional[str] = None
    data_format: Optional[str] = "jsonl"
    text_key: Optional[str] = "text"

    batch_size: Optional[int] = 1000
    limit: Optional[int] = -1  # limit the number of samples to process, for debugging

    "Save Configs"
    save_path: Optional[str] = "./data/"
    save_name: Optional[str] = "llmosaic"
    save_interval: Optional[int] = None
    logging_path: Optional[str] = "./logging/"

    def __post_init__(self):
        """
        Override properties with environment variables (or from slurm)
        """
        self.NODE_RANK = os.environ.get("NODE_RANK", "0")
        if self.NODE_RANK is None or self.NODE_RANK.strip() == '':
            self.NODE_RANK = 0
        self.NODE_RANK = int(self.NODE_RANK)
        CUDA_DEVICE = os.environ["CUDA_VISIBLE_DEVICES"]
        START_GPU = int(CUDA_DEVICE.split(",")[0])
        self.GLOBAL_RANK = (self.NODE_RANK * self.NODE_GPUS + START_GPU) // self.tp_size
        self.TOTAL_SPLIT = self.NODE_GPUS * self.N_NODES // self.tp_size
        self.logging_path = os.environ.get("logging_dir", "./logging/")

        """
        Calculate the total number of splits and check if distribution is valid
        """
        assert (
            self.NODE_GPUS % self.tp_size == 0
        ), "NODE_GPUS must be divisible by tp_size"
        assert (
            self.N_NODES * self.NODE_GPUS + START_GPU
        ) % self.tp_size == 0, f"N_NODES * NODE_GPUS + START_GPU must be divisible by tp_size; N_NODES={self.N_NODES}, NODE_GPUS={self.NODE_GPUS}, START_GPU={START_GPU}, tp_size={self.tp_size}"

        # log the config
        print(f"NODE_RANK: {self.NODE_RANK}, GLOBAL_RANK: {self.GLOBAL_RANK}, TOTAL_SPLIT: {self.TOTAL_SPLIT}")

    def get_save_dir_path(self):
        return os.path.join(self.save_path, self.save_name + f"_{self.GLOBAL_RANK}")

    @staticmethod
    def from_yaml(filepath):
        """
        Load the configuration from a YAML file
        Args:
            filepath (str): Path to the YAML file
        """
        with open(filepath, "r") as file:
            config_dict = yaml.safe_load(file)

        # Create a new instance of GentaskConfig and update its variables from the loaded data
        config = GentaskConfig()
        for key, value in config_dict.items():
            try:
                if value is None or not isinstance(value, str):
                    pass
                # if value contains env variable, get the value from the environment
                elif "$" in value:  # xxxx/$env_name/xxxxx or xxx/${env_name}/xxxx
                    env_var_pattern = re.compile(
                        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
                    )
                    env_var_name = env_var_pattern.search(value).group(
                        1
                    ) or env_var_pattern.search(value).group(2)
                    env_var_value = os.getenv(env_var_name)
                    if env_var_value is None:
                        raise ValueError(
                            f"Environment variable {env_var_name} is not set"
                        )
                    config_dict[key] = value.replace(
                        "${" + env_var_name + "}", env_var_value
                    ).replace("$" + env_var_name, env_var_value)
            except Exception as e:
                raise ValueError(f"Error processing key {key}: {e}")

            try:
                if hasattr(config, key):
                    setattr(config, key, config_dict[key])
            except Exception as e:
                raise ValueError(f"Error setting key {key}: {e}")
        return config
