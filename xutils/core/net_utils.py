import requests
from urllib import request

import xutils.core.file_utils as fu


def download_if(data_path, url, update=False):
    return fu.create_file_if(data_path,
                             lambda path: request.urlretrieve(url, path),
                             update=update)