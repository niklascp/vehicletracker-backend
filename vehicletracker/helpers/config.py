"""Helper functions for configuration"""
import yaml

def load_config(config_file):
    """Loads the given configuration file"""
    with open(config_file, "r") as file:
        config = yaml.load(file.read(), Loader = yaml.FullLoader)

    return config
