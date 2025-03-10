#!/usr/bin/env python

import subprocess
import os


def main():
    script_path = os.path.join(os.path.dirname(__file__), "clean.sh")
    subprocess.run(["bash", script_path], check=True)


if __name__ == "__main__":
    main()
