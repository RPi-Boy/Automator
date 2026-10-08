"""Joint-space pick-and-place sequencing (numpy only, no ROS imports).

PickPlaceSequencer turns one accepted target into a timed sequence of
[q1..q4, gripper] set-points and is driven by tick(dt), so it runs identically in
the ROS node, in unit tests, and offline.
"""
import numpy as np

from .config import TOOL_LENGTH, bin_table
from .kinematics import (HOME_Q, JOINT_VEL_LIMITS, IKError, forward_kinematics,
                         inverse_kinematics)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
QUINTIC_PEAK_VEL = 1.875  # max of ds/dt for s(t)=10t^3-15t^4+6t^5 on t in [0, 1]


def quintic(s):
    """Zero velocity/acceleration at both ends, s in [0, 1]."""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def segment_duration(q_a, q_b, min_time, vel_margin=0.8):
    """Longest of min_time and the time needed to respect 80% of the URDF joint velocity limits."""
    need = QUINTIC_PEAK_VEL * np.abs(np.asarray(q_b) - np.asarray(q_a)) / (vel_margin * JOINT_VEL_LIMITS)
    return float(max(min_time, need.max()))


def feasible_lift_height(xy_points, z_min=0.15, z_max=0.60, step=0.01, tool_length=0.0):
    """Lowest z at which every xy in xy_points is IK-reachable (the elbow limit blocks low, close targets)."""
    for z in np.arange(z_min, z_max + 1e-9, step):
        try:
            for x, y in xy_points:
                inverse_kinematics([x, y, z], tool_length=tool_length)
            return float(z)
        except IKError:
            continue
    raise IKError(f"no reachable lift height in [{z_min}, {z_max}] for {xy_points}")


class PickPlaceSequencer:
    """State: idle -> running -> idle. Rejects new targets while running or halted."""

    def __init__(self, place_xyz=(-0.15, 0.30, 0.03), place_by_class=None, uncertainty_threshold=0.75,
                 approach_dz=0.05, grasp_dz=0.01, release_dz=0.005, min_seg_time=1.0, dwell=0.5,
                 tool_length=TOOL_LENGTH):
        self.place_xyz = np.asarray(place_xyz, dtype=float)
        # class_id -> bin xyz (defaults to the shared 4-bin layout); classes without an entry use place_xyz
        table = bin_table() if place_by_class is None else place_by_class
        self.place_by_class = {int(k): np.asarray(v, dtype=float) for k, v in table.items()}
        self.threshold = uncertainty_threshold
        self.approach_dz, self.grasp_dz, self.release_dz = approach_dz, grasp_dz, release_dz
        self.min_seg_time, self.dwell = min_seg_time, dwell
        # distance from the wrist (ee_link origin) to the fingertips; 0 = plan for the wrist itself
        self.tool_length = tool_length
        self.q = HOME_Q.copy()
        self.gripper = GRIPPER_OPEN
        self.halted = False
        self._segments = []  # (q_start, q_end, g_start, g_end, duration)
        self._t = 0.0
        self.last_reject = ""

    @property
    def busy(self):
        return bool(self._segments)

    def halt(self):
        self.halted = True
        self._segments = []

    def clear_halt(self):
        self.halted = False

    def on_target(self, xyz, uncertainty, class_id=None):
        """Returns True if a motion was planned. Reasons for refusal land in last_reject."""
        if uncertainty > self.threshold:
            self.halt()
            self.last_reject = f"uncertainty {uncertainty:.3f} > {self.threshold}"
            return False
        if self.halted:
            self.last_reject = "halted (e-stop latched)"
            return False
        if self.busy:
            self.last_reject = "busy"
            return False
        try:
            place = self.place_by_class.get(int(class_id), self.place_xyz) if class_id is not None else self.place_xyz
            self._plan(np.asarray(xyz, dtype=float), place)
        except IKError as e:
            self.last_reject = str(e)
            return False
        self.last_reject = ""
        return True

    def _plan(self, pick, place):
        lift_z = feasible_lift_height([pick[:2], place[:2]], tool_length=self.tool_length)
        ik = lambda xyz: inverse_kinematics(xyz, tool_length=self.tool_length)  # noqa: E731
        wp = {
            "approach": ik([pick[0], pick[1], pick[2] + self.approach_dz]),
            "grasp": ik([pick[0], pick[1], pick[2] + self.grasp_dz]),
            "lift": ik([pick[0], pick[1], lift_z]),
            "transport": ik([place[0], place[1], lift_z]),
            "release": ik([place[0], place[1], place[2] + self.release_dz]),
        }
        O, C = GRIPPER_OPEN, GRIPPER_CLOSED
        plan = [
            (self.q, wp["approach"], O, O),
            (wp["approach"], wp["grasp"], O, O),
            (wp["grasp"], wp["grasp"], O, C),          # close
            (wp["grasp"], wp["lift"], C, C),
            (wp["lift"], wp["transport"], C, C),
            (wp["transport"], wp["release"], C, C),
            (wp["release"], wp["release"], C, O),      # open
            (wp["release"], wp["transport"], O, O),
            (wp["transport"], HOME_Q, O, O),
        ]
        self._segments = []
        for qa, qb, ga, gb in plan:
            hold = np.allclose(qa, qb)
            dur = self.dwell if hold else segment_duration(qa, qb, self.min_seg_time)
            self._segments.append((np.array(qa, dtype=float), np.array(qb, dtype=float), ga, gb, dur))
        self._t = 0.0

    def tick(self, dt):
        """Advance dt seconds; returns the current [q1..q4, gripper] set-point."""
        if self._segments:
            self._t += dt
            while self._segments and self._t >= self._segments[0][4]:
                qa, qb, ga, gb, dur = self._segments.pop(0)
                self._t -= dur
                self.q, self.gripper = qb, gb
            if self._segments:
                qa, qb, ga, gb, dur = self._segments[0]
                s = quintic(self._t / dur)
                self.q = qa + s * (qb - qa)
                self.gripper = ga + s * (gb - ga)
        return np.concatenate([self.q, [self.gripper]])

    def sample_path(self, dt=0.02):
        """Run the active plan to completion (testing/offline use); returns the set-point array."""
        out = []
        while self.busy:
            out.append(self.tick(dt))
        return np.array(out)


def path_min_height(path):
    return min(forward_kinematics(p[:4])[2] for p in path)
