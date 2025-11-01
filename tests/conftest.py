"""Ensure project root is importable for tests without installation.
This is a safety net; editable install is still recommended."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
