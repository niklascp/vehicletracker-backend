import yaml

def load_config(args = []):
    config_file = "configuration.yaml"
    
    with open(config_file, "r") as fd:
        config = yaml.load(fd.read(), Loader = yaml.FullLoader)

    # overrides
    for arg in args:
        [key, value] = arg.split('=', 2)
        config_level = config
        key_parts = key.split('.')
        # traverse into level
        for key_part in key_parts[:-1]:
            if not key_part in config_level:
                config_level[key_part] = {}
            config_level = config_level[key_part]
        # override last part
        config_level[key_parts[-1]] = value

    return config
