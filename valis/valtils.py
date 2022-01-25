"""Various functions used in serval modules

"""

import re
import warnings
import os
from colorama import init as color_init
from colorama import Fore, Style

color_init()


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
