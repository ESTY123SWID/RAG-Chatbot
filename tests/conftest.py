import sys
from pathlib import Path

# מאפשר לייבא את מודולי הפרויקט מתוך תיקיית tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
