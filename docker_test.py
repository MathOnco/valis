# print("hello world")

#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name of person to greet")

args = parser.parse_args()

print(f"Hello there, {args.name}!")

import os
print(os.getcwd(), os.path.exists(os.getcwd()), os.path.exists("/Users/gatenbcd/Dropbox/Documents/BCI-EvoCa2/chandler/CycIF_example/slides/CycIF_Example1_K3"), os.listdir(os.getcwd()))
p = os.path.join(os.getcwd(), "Users/gatenbcd/Dropbox/Documents")
print(p, os.listdir(p))

os.path.exists("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docs/_images/annotation_transfer.png")
# "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_test.py:main.py"


# docker run -it --rm -v "$PWD:$PWD" -w "$PWD" valis docker_test.py "yo"
# docker run -it --rm -v "$PWD:$PWD" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_test.py "yo"
# docker run -it --rm -v "USER:$PWD" -w "UID$PWD" valis docker_test.py "yo"