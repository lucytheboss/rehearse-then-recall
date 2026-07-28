"""Notebook initial setup — bundles fixing the project root, loading .env, and
checking API key status into one function.

In the notebook's first cell, add the project root to sys.path so this
module can be imported, then call setup_project() once to handle the rest
(fixing the working directory, loading .env, printing whether the key was
recognized).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def find_project_root(marker: str = "src") -> Path:
    """Walks up from the current working directory to find the point where a marker (default "src") directory exists."""
    root = Path.cwd()
    while not (root / marker).exists() and root != root.parent:
        root = root.parent
    return root


def setup_project(api_key_env: str = "NVIDIA_NIM_API_KEY") -> Path:
    """Handles a notebook's early boilerplate in one call.

    1) Finds the project root and fixes the working directory and sys.path
    2) Loads .env — uses find_dotenv(usecwd=True) to search from cwd instead
       of the call stack (the default find_dotenv() sometimes fails under a
       notebook kernel's exec-based execution, so this is more reliable)
    3) Prints whether the API key was recognized, for an immediate visual check

    Returns the project root path.
    """
    root = find_project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    loaded_env_path = load_dotenv(find_dotenv(usecwd=True))

    print("project root:", root)
    print(".env loaded:", "success ✅" if loaded_env_path else "n/a (no .env file)")

    if os.environ.get(api_key_env):
        print(f"{api_key_env}: set ✅")
    else:
        print(f"{api_key_env}: missing ❌ — copy .env.example to .env and fill in the key")

    return root
