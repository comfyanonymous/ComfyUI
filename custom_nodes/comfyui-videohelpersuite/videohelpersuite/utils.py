import hashlib
import os
from typing import Iterable
import shutil
import subprocess
import re
import time
from collections.abc import Mapping
from typing import Union
import functools
import torch
from torch import Tensor

import server
from .logger import logger
import folder_paths

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192

ENCODE_ARGS = ("utf-8", 'backslashreplace')

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types=allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types
imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

class ContainsAll(dict):
    def __contains__(self, other):
        return True
    def __getitem__(self, key):
        return super().get(key, (None, {}))

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            #Evaluation of suitability isn't required, can take sole option
            #to reduce startup time
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)
gifski_path = os.environ.get("VHS_GIFSKI", None)
if gifski_path is None:
    gifski_path = os.environ.get("JOV_GIFSKI", None)
    if gifski_path is None:
        gifski_path = shutil.which("gifski")
ytdl_path = os.environ.get("VHS_YTDL", None) or shutil.which('yt-dlp') \
        or shutil.which('youtube-dl')
download_history = {}
def try_download_video(url):
    if ytdl_path is None:
        return None
    if url in download_history:
        return download_history[url]
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    #Format information could be added to only download audio for Load Audio,
    #but this gets hairy if same url is also used for video.
    #Best to just always keep defaults
    #dl_format = ['-f', 'ba'] if is_audio else []
    try:
        res = subprocess.run([ytdl_path, "--print", "after_move:filepath",
                              "-P", folder_paths.get_temp_directory(), url],
                             capture_output=True, check=True)
        #strip newline
        file = res.stdout.decode(*ENCODE_ARGS)[:-1]
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the yt-dl process:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
        file = None
    download_history[url] = file
    return file

def is_safe_path(path, strict=False):
    if "VHS_STRICT_PATHS" not in os.environ and not strict:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = strip_path(directory)
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

prompt_queue = server.PromptServer.instance.prompt_queue
def requeue_workflow_unchecked():
    """Requeues the current workflow without checking for multiple requeues"""
    currently_running = prompt_queue.currently_running
    (_, _, prompt, extra_data, outputs_to_execute) = next(iter(currently_running.values()))

    #Ensure batch_managers are marked stale
    prompt = prompt.copy()
    for uid in prompt:
        if prompt[uid]['class_type'] == 'VHS_BatchManager':
            prompt[uid]['inputs']['requeue'] = prompt[uid]['inputs'].get('requeue',0)+1

    #execution.py has guards for concurrency, but server doesn't.
    #TODO: Check that this won't be an issue
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(server.uuid.uuid4())
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))

requeue_guard = [None, 0, 0, {}]
def requeue_workflow(requeue_required=(-1,True)):
    assert(len(prompt_queue.currently_running) == 1)
    global requeue_guard
    (run_number, _, prompt, _, _) = next(iter(prompt_queue.currently_running.values()))
    if requeue_guard[0] != run_number:
        #Calculate a count of how many outputs are managed by a batch manager
        managed_outputs=0
        for bm_uid in prompt:
            if prompt[bm_uid]['class_type'] == 'VHS_BatchManager':
                for output_uid in prompt:
                    if prompt[output_uid]['class_type'] in ["VHS_VideoCombine"]:
                        for inp in prompt[output_uid]['inputs'].values():
                            if inp == [bm_uid, 0]:
                                managed_outputs+=1
        requeue_guard = [run_number, 0, managed_outputs, {}]
    requeue_guard[1] = requeue_guard[1]+1
    requeue_guard[3][requeue_required[0]] = requeue_required[1]
    if requeue_guard[1] == requeue_guard[2] and max(requeue_guard[3].values()):
        requeue_workflow_unchecked()

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        #TODO: scan for sample rate and maintain
        res =  subprocess.run(args + ["-f", "f32le", "-"],
                              capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        #NOTE: Just throwing an error for other channel types right now
        #Will deal with issues if they come
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time=start_time
        self.duration=duration
        self._dict=None
    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]
    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)
    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)
def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)

def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_sequence(path):
    #Check if path is a valid ffmpeg sequence that points to at least one file
    (path, file) = os.path.split(path)
    if not os.path.isdir(path):
        return False
    match = re.search('%0?\\d+d', file)
    if not match:
        return False
    seq = match.group()
    if seq == '%d':
        seq = '\\\\d+'
    else:
        seq = '\\\\d{%s}' % seq[1:-1]
    file_matcher = re.compile(re.sub('%0?\\d+d', seq, file))
    for file in os.listdir(path):
        if file_matcher.fullmatch(file):
            return True
    return False

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path
def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    if not os.path.isfile(path):
        return "DNE"
    return calculate_file_hash(strip_path(path))


def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return "Invalid file path: {}".format(path)
    return is_safe_path(path)


def validate_index(index: int, length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False) -> int:
    # if part of range, do nothing
    if is_range:
        return index
    # otherwise, validate index
    # validate not out of range - only when latent_count is passed in
    if length > 0 and index > length-1 and not allow_missing:
        raise IndexError(f"Index '{index}' out of range for {length} item(s).")
    # if negative, validate not out of range
    if index < 0:
        if not allow_negative:
            raise IndexError(f"Negative indeces not allowed, but was '{index}'.")
        conv_index = length+index
        if conv_index < 0 and not allow_missing:
            raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for {length} item(s).")
        index = conv_index
    return index


def convert_to_index_int(raw_index: str, length: int=0, is_range: bool=False, allow_negative=False, allow_missing=False) -> int:
    try:
        return validate_index(int(raw_index), length=length, is_range=is_range, allow_negative=allow_negative, allow_missing=allow_missing)
    except ValueError as e:
        raise ValueError(f"Index '{raw_index}' must be an integer.", e)


def convert_str_to_indexes(indexes_str: str, length: int=0, allow_missing=False) -> list[int]:
    if not indexes_str:
        return []
    int_indexes = list(range(0, length))
    allow_negative = length > 0
    chosen_indexes = []
    # parse string - allow positive ints, negative ints, and ranges separated by ':'
    groups = indexes_str.split(",")
    groups = [g.strip() for g in groups]
    for g in groups:
        # parse range of indeces (e.g. 2:16)
        if ':' in g:
            index_range = g.split(":", 2)
            index_range = [r.strip() for r in index_range]

            start_index = index_range[0]
            if len(start_index) > 0:
                start_index = convert_to_index_int(start_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing)
            else:
                start_index = 0
            end_index = index_range[1]
            if len(end_index) > 0:
                end_index = convert_to_index_int(end_index, length=length, is_range=True, allow_negative=allow_negative, allow_missing=allow_missing)
            else:
                end_index = length
            # support step as well, to allow things like reversing, every-other, etc.
            step = 1
            if len(index_range) > 2:
                step = index_range[2]
                if len(step) > 0:
                    step = convert_to_index_int(step, length=length, is_range=True, allow_negative=True, allow_missing=True)
                else:
                    step = 1
            # if latents were passed in, base indeces on known latent count
            if len(int_indexes) > 0:
                chosen_indexes.extend(int_indexes[start_index:end_index][::step])
            # otherwise, assume indeces are valid
            else:
                chosen_indexes.extend(list(range(start_index, end_index, step)))
        # parse individual indeces
        else:
            chosen_indexes.append(convert_to_index_int(g, length=length, allow_negative=allow_negative, allow_missing=allow_missing))
    return chosen_indexes


def select_indexes(input_obj: Union[Tensor, list], idxs: list):
    if type(input_obj) == Tensor:
        return input_obj[idxs]
    else:
        return [input_obj[i] for i in idxs]

def merge_filter_args(args, ftype="-vf"):
    #TODO This doesn't account for filter_complex
    #Will likely need to convert all filters to filter complex in the future
    #But that requires source/output deduplication
    try:
        start_index = args.index(ftype)+1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += ',' + args[index+1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass

def select_indexes_from_str(input_obj: Union[Tensor, list], indexes: str, err_if_missing=True, err_if_empty=True):
    real_idxs = convert_str_to_indexes(indexes, len(input_obj), allow_missing=not err_if_missing)
    if err_if_empty and len(real_idxs) == 0:
        raise Exception(f"Nothing was selected based on indexes found in '{indexes}'.")
    return select_indexes(input_obj, real_idxs)

def hook(obj, attr):
    def dec(f):
        f = functools.update_wrapper(f, getattr(obj,attr))
        setattr(obj,attr,f)
        return f
    return dec

def cached(duration):
    def dec(f):
        cached_ret = None
        cache_time = 0
        def cached_func():
            nonlocal cache_time, cached_ret
            if time.time() > cache_time + duration or cached_ret is None:
                cache_time = time.time()
                cached_ret = f()
            return cached_ret
        return cached_func
    return dec
