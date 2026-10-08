"""Locate the Automator repo root (perception/ and sim/ live there, outside this ROS package)."""
import os


def find_project_root():
    """AUTOMATOR_ROOT env var, else walk up from this file (works with `colcon build --symlink-install`)."""
    env = os.environ.get('AUTOMATOR_ROOT')
    candidates = [env] if env else []
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidates.append(d)
        d = os.path.dirname(d)
    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'perception', 'detector.py')):
            return c
    raise RuntimeError('Cannot find the Automator repo (perception/). Export AUTOMATOR_ROOT=/path/to/Automator '
                       'or build with `colcon build --symlink-install`.')
