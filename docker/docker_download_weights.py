"""
Download pytorch model weights when building Docker container.
Weights get downloaded during initialization
"""

import torch
import kornia
from valis import feature_detectors, feature_matcher#, non_rigid_registrars



print("Downloading DiskFD weights")
disk_fd = feature_detectors.DiskFD()
print("Downloading DeDoDeFD weights")
dedode_fd = feature_detectors.DeDoDeFD()

disk_matcher = feature_matcher.LightGlueMatcher(disk_fd)
dedode_matcher = feature_matcher.LightGlueMatcher(dedode_fd)

# raft_nr = non_rigid_registrars.RAFTWarper()
