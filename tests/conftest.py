import sys
from pathlib import Path

# make `config` and `residlstm` importable when pytest is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
