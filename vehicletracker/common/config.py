import yaml

def load_config():
    config_file = "configuration.yaml"
    with open(config_file, "r") as fd:
        config = yaml.load(fd.read(), Loader = yaml.FullLoader)
    return config
