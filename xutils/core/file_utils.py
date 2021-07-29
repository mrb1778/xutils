import glob
import os
import shutil
import tarfile
import tempfile

from urllib.request import urlopen

from pathlib import Path


def path_inspect(path, search="*", trim_path=True):
    for file_name in Path(path).rglob(search):
        print(str(file_name)[len(path) + 1:] if trim_path else file_name)


def list_files(path):
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        files.extend(file_names)
        break

    return files


def get_files(path, name_only=False):
    files = list_files(path)
    return [x if name_only else os.path.join(path, x) for x in files]


def get_file_name(path, strip_extension=False):
    name = Path(path).name
    if strip_extension:
        return name.split(".")[0]
    else:
        return name


def exists(path, msg):
    assert os.path.exists(path), msg


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_and_unzip(url):
    with tempfile.TemporaryFile() as tmp:
        shutil.copyfileobj(urlopen(url), tmp)
        tmp.seek(0)
        tar = tarfile.open(fileobj=tmp)
        tar.extractall()
        tar.close()


def count_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
    return count


def count_directories(directory):
    return len(glob.glob(directory + "/*"))


def create_file_if(path, create_fn, update=False):
    if not os.path.isfile(path) or update:
        create_parent_dirs(path)
        return create_fn(path)
    else:
        return path


def create_parent_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_difference(file1, file2, print_results=True):
    check = {}
    for file in [file1, file2]:
        with open(file, 'r') as f:
            check[file] = []
            for line in f:
                check[file].append(line)
    diff = set(check[file1]) - set(check[file2])
    if print_results:
        for line in diff:
            print(line.rstrip())

    return diff


FILENAME_SAFE_CHARS = (' ', '.', '_')


def to_file_name(file_name, replace_with="_"):
    return replace_with.join(c for c in file_name if c.isalnum() or c in FILENAME_SAFE_CHARS).rstrip()
