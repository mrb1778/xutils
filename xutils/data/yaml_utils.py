import yaml


def is_yaml(extension: str) -> bool:
    return extension in (".yml", ".yaml")


def read_file(path):
    with open(path) as file:
        return yaml.safe_load(file)



