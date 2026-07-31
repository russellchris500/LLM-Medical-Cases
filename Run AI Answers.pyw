"""Double-click launcher for Run AI Answers (no console window)."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import hub_runner

hub_runner.main()
