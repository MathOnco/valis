import torch
import kornia

import re
import os
import multiprocessing
from colorama import init as color_init
from colorama import Fore, Style
import functools
import pyvips
import warnings
import contextlib
from collections import defaultdict
import platform
import subprocess


color_init()


def print_warning(msg, warning_type=UserWarning, rgb=Fore.YELLOW, traceback_msg=None):
    """Print warning message with color
    """
    warning_msg = f"{rgb}{msg}{Style.RESET_ALL}"
    if warning_type is None:
        print(warning_msg)
    else:
        warnings.simplefilter('always', warning_type)
        warnings.warn(warning_msg, warning_type)

    if traceback_msg is not None:
        traceback_msg_rgb = f"{rgb}{traceback_msg}{Style.RESET_ALL}"
        print(traceback_msg_rgb)

def deprecated_args(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco

def rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError('{} received both {} and {}'.format(
                    func_name, alias, new))

            msg = f'{alias} is deprecated; use {new} instead'
            print_warning(msg, DeprecationWarning)

            kwargs[new] = kwargs.pop(alias)


@contextlib.contextmanager
def HiddenPrints():
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        yield


def pad_strings(string_list, side="r"):
    """
    side : string
        Which side to add the padding to
    """
    if side.lower().startswith("r"):
        pad_fxn = "ljust"
    else:
        pad_fxn = "rjust"

    max_chr = max([len(x) for x in string_list])
    padded_strings = [x.__getattribute__(pad_fxn)(max_chr) for x in string_list]

    return padded_strings

def check_m1_mac():
    is_mac_m1 = False
    if platform.system() == "Darwin":
        cpu_kind = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode('utf-8')
        if cpu_kind.startswith("Apple M1"):
            is_mac_m1 = True

    return is_mac_m1


from . import slide_tools # Put import here to avoid circular imports
def get_name(f):

    fonly = os.path.split(f)[1]
    img_extension = slide_tools.get_slide_extension(fonly)
    if img_extension is None or len(img_extension) == 0:
        # Extension already removed, so this is probably already a name
        img_name = fonly
    else:
        img_name = img_extension.join(fonly.split(img_extension)[:-1])

    return img_name


def levenshtein_d(str1, str2):
    # Get the lengths of the input strings
    m = len(str1)
    n = len(str2)

    # Initialize two rows for dynamic programming
    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = i

        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Choose the minimum cost operation
                curr_row[j] = 1 + min(
                    curr_row[j - 1], # Insert
                    prev_row[j],	 # Remove
                    prev_row[j - 1] # Replace
                )

        # Update the previous row with the current row
        prev_row = curr_row.copy()

    # The final element in the last row contains the Levenshtein distance
    return curr_row[n]


def sort_nicely(l):
    """Sort the given list in the way that humans expect.
    """
    l.sort(key=lambda s: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)])


def get_elapsed_time_string(elapsed_time, rounding=3):
    """Format elapsed time

    Parameters
    ----------
    elapsed_time : float
        Elapsed time in seconds

    rounding : int
        Number of decimal places to round

    Returns
    -------
    scaled_time : float
        Scaled amount of elapsed time

    time_unit : str
        Time unit, either seconds, minutes, or hours

    """

    if elapsed_time < 60:
        scaled_time = elapsed_time
        time_unit = "seconds"

    elif 60 <= elapsed_time < 60 ** 2:
        scaled_time = elapsed_time / 60
        time_unit = "minutes"

    else:
        scaled_time = elapsed_time / (60 ** 2)
        time_unit = "hours"

    scaled_time = round(scaled_time, rounding)

    return scaled_time, time_unit


def get_vips_version():
    try:
        v = f"{pyvips.vips_lib.VIPS_MAJOR_VERSION}.{pyvips.vips_lib.VIPS_MINOR_VERSION}.{pyvips.vips_lib.VIPS_MICRO_VERSION}"
    except AttributeError:
        v = ".".join([str(pyvips.vips_lib.vips_version(i)) for i in range(3)])

    return v


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_ncpus_available():
    ncpus = 2  # returning 2 by default as code assumes ncpus > 1 (or that packages gracefully handle scheduling 0 jobs/threads)

    if hasattr(os, "sched_getaffinity"):
        ncpus = len(os.sched_getaffinity(0))
    elif hasattr(multiprocessing, "cpu_count"):
        ncpus = multiprocessing.cpu_count()

    return int(ncpus)

