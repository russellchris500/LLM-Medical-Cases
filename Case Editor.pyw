"""Double-click launcher for Program 1 - Case Editor (for providers).

The .pyw ending makes Windows open the window WITHOUT a black console
window behind it. Keep this file in the same folder as the program files.
"""

import os
import sys

folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)          # data files live next to the programs
sys.path.insert(0, folder)

import case_editor

raise SystemExit(case_editor.main())
