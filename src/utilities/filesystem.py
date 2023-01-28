import os
import yaml
import zipfile
import tarfile
import h5py
import pickle
from scipy.io import wavfile


def save_dict(dictionary, path, format = "yaml"):
    if format == "pickle":
        pickle_save(dictionary, path)
    elif format == "yaml":
        yaml_save(dictionary, path)
    else:
        raise NotImplementedError("Unknown format " + format)

def load_dict(path, format = "yaml"):
    if format == "pickle":
        dictionary = pickle_load(path)
    elif format == "yaml":
        dictionary = yaml_load(path)
    else:
        raise NotImplementedError("Unknown format " + format)
    return dictionary

def yaml_save(dictionary, path):
    with open(path, "w") as f:
        yaml.dump(dictionary, f)

def yaml_load(path):
    with open(path) as f:
        dictionary = yaml.safe_load(f)
    return dictionary

def pickle_save(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
def pickle_load(path):
    with open(path, "rb") as f:
        loaded_dictionary = pickle.load(f)
    return loaded_dictionary


def save_ndarray(file, path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=file)

def load_ndarray(path):
    with h5py.File(path, "r") as f:
        out = f["data"][()]
    return out

def safe_mkdir(path):
    try:
        os.mkdir(path)
        return 0
    except:
        return 1

def unzip_to_directory(zip_path, dir_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)

def untar_to_directory(tar_path, dir_path):
    with tarfile.open(tar_path) as tar_ref:
        tar_ref.extractall(dir_path, numeric_owner=True) # specify which folder to extract to

def unpack_to_directory(pack_path, dir_path):
    if zipfile.is_zipfile(pack_path):
        unzip_to_directory(pack_path, dir_path)
    elif tarfile.is_tarfile(pack_path):
        untar_to_directory(pack_path, dir_path)
    else:
        raise NotImplementedError(f"Unknown package type, only 'zip' or 'tar' are possible ")

def is_dir_used(dir_path):
    try:
        if len(os.listdir(dir_path)) > 0:
            return True
        return False
    except:
        return False

def safe_mkdir_path(dir_path):
    path = os.path.normpath(dir_path).split(os.sep)
    for folder_id in range(len(path)):
        safe_mkdir(os.path.join(*path[:folder_id+1]).replace(":", ":\\"))

def safe_remove(path):
    try:
        os.remove(path)
        return 0
    except:
        return 1

def read_audio_file(file_path):
    frequency_sampling, audio_signal = wavfile.read(file_path)
    return frequency_sampling, audio_signal


