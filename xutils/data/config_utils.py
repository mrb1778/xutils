import xutils.core.file_utils as fu

import xutils.data.json_utils as ju
import xutils.data.yaml_utils as yu


def read_file(path):
    extension = fu.extension(path)
    if ju.is_json(extension):
        return ju.read_file(path)
    elif yu.is_yaml(extension):
        return yu.read_file(path)
