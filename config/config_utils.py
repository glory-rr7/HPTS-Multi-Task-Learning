import os
import yaml


def LoadConfig():

    yaml_file_name = "config.yaml"

    yaml_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), yaml_file_name)

    # LOAD CONFIGURATIONS
    with open(yaml_file_path, encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


