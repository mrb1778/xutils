import glob

import os
import shutil
import tarfile
import tempfile
from typing import Union, Optional, List, Set

from urllib.request import urlopen

from pathlib import Path, PurePath
from datetime import datetime


def path_inspect(path: str, search="*", trim_path=True) -> None:
    for name in Path(path).rglob(search):
        print(str(name)[len(path) + 1:] if trim_path else name)


def list_paths(path: str,
               file_extension: str = None,
               sort_name: bool = True,
               sort_size: bool = False,
               sort_updated: bool = False,
               recursive: bool = False,
               only_directories: bool = False,
               just_name: bool = False) -> Optional[List[str]]:
    paths = glob.glob(os.path.join(path, "*" if file_extension is None else f"*.{file_extension}"),
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


def get_first_path(path: str,
                   file_extension: str = None,
                   sort_name: bool = True,
                   sort_size: bool = False,
                   sort_updated: bool = False,
                   recursive: bool = False,
                   only_directories: bool = False,
                   just_name: bool = False) -> str:
    paths = list_paths(path=path,
                       file_extension=file_extension,
                       sort_name=sort_name,
                       sort_size=sort_size,
                       sort_updated=sort_updated,
                       recursive=recursive,
                       only_directories=only_directories,
                       just_name=just_name)
    return paths[0] if len(paths) else None


def file_name(path: str, strip_extension: bool = False) -> str:
    path = Path(path)
    if strip_extension:
        return path.stem
    else:
        return path.name


def extension(path: str) -> str:
    return Path(path).suffix


def exists(path: str) -> bool:
    return Path(path).exists()


def ensure_exists(path: str) -> bool:
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    return False


def delete(path: str) -> None:
    Path(path).unlink()


def delete_files(path: str) -> None:
    for f in Path(path).glob("*"):
        if f.is_file():
            f.unlink()


def delete_all(path: str) -> None:
    for path in Path(path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def move(path: str, to: str):
    ensure_exists(to)
    path = Path(path)
    # return str(path.rename(os.path.join(Path(to), path.name))) --> permission problems
    return shutil.move(path, os.path.join(Path(to), path.name))


def copy(path: str, to: str, with_prefix: bool = False):
    ensure_exists(to)
    path = Path(path)
    if with_prefix:
        for p in path.parent.glob(f"{path.name}*"):
            if p.is_file():
                copy(str(p), to, with_prefix=False)
    else:
        return shutil.copy2(str(path), to)


def download_and_unzip(url: str) -> None:
    with tempfile.TemporaryFile() as tmp:
        shutil.copyfileobj(urlopen(url), tmp)
        tmp.seek(0)
        tar = tarfile.open(fileobj=tmp)
        tar.extractall()
        tar.close()


def count_files(directory: str) -> int:
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
    return count


def count_directories(directory: str) -> int:
    return len(glob.glob(directory + "/*"))


def create_file_if(path: str,
                   create_fn,
                   update_if_older_than: int = None,
                   update=False):
    if update or \
            not os.path.isfile(path) or \
            (update_if_older_than is not None and modified_days_ago(path) > update_if_older_than):
        create_parent_dirs(path)
        return create_fn(path)
    else:
        return path


def create_parent_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_dirs(path: str) -> None:
    shutil.rmtree(path)


def get_difference(file1: str, file2: str, print_results: bool = True) -> Set:
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


def to_file_name(file_name: str, replace_with: str = "_"):
    return replace_with.join(c for c in file_name if c.isalnum() or c in FILENAME_SAFE_CHARS).rstrip()


def script_path(file: str = None):
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


def modified(path: str) -> float:
    return Path(path).stat().st_mtime if path else 0


def modified_days_ago(path: str) -> int:
    return (datetime.now() - datetime.fromtimestamp(modified(path))).days


def modified_after(after, before) -> bool:
    return modified(after) > modified(before)
