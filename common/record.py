import os
import sys
import subprocess


def record_setting(out):
    """Record scripts and commandline arguments"""
    os.makedirs(out, exist_ok=True)  
    subprocess.call("cp *.py %s" % out, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")
