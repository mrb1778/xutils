import glob

import os
import shutil
import tarfile
import tempfile

from urllib.request import urlopen

from pathlib import Path, PurePath


def path_inspect(path, search="*", trim_path=True):
    for file_name in Path(path).rglob(search):
        print(str(file_name)[len(path) + 1:] if trim_path else file_name)


def list_paths(path,
               extension=None,
               sort_name=True,
               sort_size=False,
               sort_updated=False,
               recursive=False,
               only_directories=False,
               just_name=False):
    paths = glob.glob(os.path.join(path, "*" if extension is None else f"*.{extension}"),
                      recursive=recursive)
    if only_directories:
        paths = [path for path in paths if os.path.isdir(path)]

    if sort_name or sort_size or sort_updated:
        paths = sorted(paths, key=lambda x: (
            x if sort_name else None,
            os.path.getsize(x) if sort_size else None,
            os.path.getmtime(x) if sort_updated else None,
        ))

    if just_name:
        paths = [PurePath(path).name for path in paths]
    return paths


def get_file_name(path, strip_extension=False):
    name = Path(path).name
    if strip_extension:
        return name.split(".")[0]
    else:
        return name


def exists(path):
    return os.path.exists(path)


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete(path):
    Path(path).unlink()


def delete_files(path):
    for f in Path(path).glob("*"):
        if f.is_file():
            f.unlink()


def delete_all(path):
    for path in Path(path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def move(path, to):
    ensure_exists(to)
    path = Path(path)
    # return str(path.rename(os.path.join(Path(to), path.name))) --> permission problems
    return shutil.move(path, os.path.join(Path(to), path.name))


def copy(path, to, with_prefix=False):
    ensure_exists(to)

    path = Path(path)
    if with_prefix:
        for p in path.parent.glob(f"{path.name}*"):
            if p.is_file():
                copy(p, to, with_prefix=False)
    else:
        return shutil.copy(str(path), os.path.join(Path(to), path.name))


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


def create_dirs(path):
    os.makedirs(path, exist_ok=True)


def remove_dirs(path):
    shutil.rmtree(path)


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


def script_path(file=None):
    return os.path.dirname(
        os.path.realpath(
            __file__ if file is None else file))


def add_seekable_to_file(f):
    """
    If file f does not has seekable function -
    add seekable function that will always return true
    Args:
        f: the file
    Returns: the file f with seekable function

    """
    if not hasattr(f, "seekable"):
        # AFAICT all the filetypes that STF wraps can seek
        f.seekable = lambda: True

    return f
