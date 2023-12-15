import os
import ntpath
import shutil
from pathlib import Path
import glob






def create_replace_existing_path(path, force=False, create=True, **kwargs):
    """
    This function, based on the 'force' parameter, deletes an existing path.
    Then, based on the 'create' parameter, if the path points to a folder, it creates a new one.

    Parameters
    ----------
    path: string
        Path of the file/folder to replace.
    force: bool, default False
        A boolean value to erase or not an existing path.
    create: bool, default True
        A boolean value to create a new folder when deleted.
    kwargs:
        Keyword arguments added for compatibility.

    Returns
    -------
        None

    """

    # CHECK IF THE PATH EXISTS
    exists = os.path.exists(path)

    _, file_name = os.path.split(path)

    is_file = "." in file_name
    if exists and not is_file:
        if is_file and exists:
            os.remove(path)
        else:
            if force:
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)
    else:
        os.makedirs(path)

def create_dir(outdir): # function to create directory
    if not os.path.exists(outdir):
        Path(outdir).mkdir(parents=True, exist_ok=True) # with parents 'True' creates all tree/nested folder

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def listdir_nohidden_with_path(path):
    return glob.glob(os.path.join(path, '*'))

def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders

def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))

def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]

def create_path(*path_list, f=None):
    f = path_list[0]
    for i in range(1, len(path_list)):
        path = str(path_list[i])
        f = os.path.join(f, path)
    return f

def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def create_run_dir_local(run_dir_root) -> str:
    """Create a new run dir with increasing ID number at the start."""

    if not os.path.exists(run_dir_root):
        print("Creating the run dir root: {}".format(run_dir_root))
        os.makedirs(run_dir_root)

    run_id = get_next_run_id_local(run_dir_root)
    run_name = "{0:05d}".format(run_id)
    run_dir = os.path.join(run_dir_root, run_name)

    if os.path.exists(run_dir):
        raise RuntimeError("The run dir already exists! ({0})".format(run_dir))

    print("Creating the run dir: {}".format(run_dir))
    os.makedirs(run_dir)

    return run_dir

def get_next_run_id_local(run_dir_root: str, module_name: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    import re
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name]
    dir_names = []
    for d in os.listdir(run_dir_root):
        if not 'configuration.yaml' in d and not 'log.txt' in d and not 'src' in d:
            try:
                if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name:
                    dir_names.append(d)
            except IndexError:
                if os.path.isdir(os.path.join(run_dir_root, d)):
                    dir_names.append(d)

    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 1

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id