"""Various functions used in serval modules

"""
import sys
import re
import warnings
import os
from colorama import init as color_init
from colorama import Fore, Style
import functools
import warnings

color_init()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def get_name(f):
    """
    To get an object's name, remove image type extension from filename
    """
    if re.search("\.", f) is None:
        # Extension already removed
        return f

    f = os.path.split(f)[-1]

    if f.endswith(".ome.tiff") or f.endswith(".ome.tif"):
        back_slice_idx = 2
    else:
        back_slice_idx = 1

    img_name = "".join([".".join(f.split(".")[:-back_slice_idx])])

    return img_name


def print_warning(msg, warning_type=UserWarning, rgb=Fore.RED):
    """Print warning message with color
    """
    warning_msg = f"{rgb}{msg}{Style.RESET_ALL}"
    if warning_type is None:
        print(warning_msg)
    else:
        warnings.simplefilter('always', UserWarning)
        warnings.warn(warning_msg, warning_type)


def get_elapsed_time_string(elapsed_time, rounding=3):
    """Format elpased time

    Parameters
    ----------
    elapsed_time : float
        Elapsed time in seconds

    rounding : int
        Number of decimal places to round

    Returns
    -------
    processing_time : float
        Scaled amount elapsed time

    processing_time_unit : str
        Time unit, either seconds, minutes, or hours

    """

    if elapsed_time < 60:
        processing_time = elapsed_time
        processing_time_unit = "seconds"

    elif 60 <= elapsed_time < 60**2:
        processing_time = elapsed_time/60
        processing_time_unit = "minutes"

    else:
        processing_time = elapsed_time/(60**2)
        processing_time_unit = "hours"

    processing_time = round(processing_time, rounding)

    return processing_time, processing_time_unit


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

            msg = '{} is deprecated; use {}'.format(alias, new)
            print_warning(msg, DeprecationWarning)

            kwargs[new] = kwargs.pop(alias)

# Example of using deprecated_args decoraator
@deprecated_args(old_arg="new_arg")
def test_dep_func(new_arg):
    print(new_arg)
