# Project setup
## source directories


# Rigid registration
## General info

## Options
### Feature detectors
* List all
```
from valis import feature_detectors
import inspect

feature_detector_names = []
base_fd_class = feature_detectors.FeatureDD
for name, obj in inspect.getmembers(feature_detectors):
    if inspect.isclass(obj):
        if issubclass(obj, base_fd_class):
            feature_detector_names.append(name)
```

* Default = VggFD


# Non-rigid registration
## General info
Optional
