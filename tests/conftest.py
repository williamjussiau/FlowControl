import sys
from pathlib import Path

# Add src/ to path so tests can import without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
