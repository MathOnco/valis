"""Test that all dependencies have been installed

Won't try to import pyvips, because it requires system level libraries
which may not already be installed

"""

import scipy
import numba
import numpy
import skimage
import sklearn
import cv2
import sitk
import matplotlib.pyplot
import tqdm
import pandas
import fastcluster
import joblib
import PIL
from bs4 import BeautifulSoup
import ome_types
import jpype
import bioformats_jar
import shapely
import interpolation
import colorama
import colour
