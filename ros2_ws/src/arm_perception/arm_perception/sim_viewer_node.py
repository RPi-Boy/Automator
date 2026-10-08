"""Demo window: simulation view | perception view, with live status and keyboard e-stop controls.

Keys (click the window first):  e = send e-stop   r = reset e-stop   q = quit
If there is no display (DISPLAY unset, e.g. over plain SSH) the window is skipped and, if `save_video`
is set, the same view is written to an mp4 instead.

Subscribes: /sim/spectator, /perception/debug, /camera/image_raw (fallback), /arm_target, /arm_busy,
            /estop, /esp32/status, /joint_commands
Publishes : /estop (Bool) and /estop_reset (Bool) on key presses
"""
import os
import time

import numpy as np

from .overlay import GREEN, RED, WHITE, compose_view, decode_esp32_status

try:
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32MultiArray, Int32
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object


def image_to_rgb(msg):
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = row[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1].copy() if msg.encoding == 'bgr8' else img.copy()


class SimViewerNode(Node):

    def __init__(self):
        super().__init__('sim_viewer_node')
        self.declare_parameter('save_video', '')
        self.declare_parameter('show_window', True)
        self.show = bool(self.get_parameter('show_window').value) and bool(os.environ.get('DISPLAY'))
        self.video_path = self.get_parameter('save_video').value
        self.writer = None
        self.spectator = self.debug = self.raw = None
        self.target = None
        self.busy = False
        self.estop = False
        self.esp = None
        self.last_cmd = None
        self.t_start = time.monotonic()
        self.t_esp = 0.0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.estop_pub = self.create_publisher(Bool, '/estop', latched)
        self.reset_pub = self.create_publisher(Bool, '/estop_reset', 10)
        self.create_subscription(Image, '/sim/spectator', lambda m: setattr(self, 'spectator', image_to_rgb(m)), 1)
        self.create_subscription(Image, '/perception/debug', lambda m: setattr(self, 'debug', image_to_rgb(m)), 1)
        self.create_subscription(Image, '/camera/image_raw', lambda m: setattr(self, 'raw', image_to_rgb(m)), 1)
        self.create_subscription(Float32MultiArray, '/arm_target', lambda m: setattr(self, 'target', list(m.data)), 5)
        self.create_subscription(Float32MultiArray, '/joint_commands', lambda m: setattr(self, 'last_cmd', list(m.data)), 1)
        self.create_subscription(Bool, '/arm_busy', lambda m: setattr(self, 'busy', m.data), 1)
        self.create_subscription(Bool, '/estop', lambda m: setattr(self, 'estop', m.data), latched)
        self.create_subscription(Int32, '/esp32/status', self.on_esp, 5)
        self.create_timer(1.0 / 15.0, self.on_timer)
        if not self.show:
            self.get_logger().warn('No display available (or show_window:=false): not opening a window.'
                                   + (f' Writing {self.video_path}' if self.video_path else ' Set save_video:=out.mp4 to record.'))
        else:
            cv2.namedWindow('Automator demo', cv2.WINDOW_NORMAL)

    def on_esp(self, msg):
        self.esp = decode_esp32_status(int(msg.data))
        self.t_esp = time.monotonic()

    def status_lines(self):
        now = time.monotonic()
        if self.estop:
            l1 = ('E-STOP ACTIVE - arm halted.   press r to reset', RED)
        else:
            l1 = ('ARM MOVING' if self.busy else 'ARM IDLE - perception running', GREEN)
        if self.target and len(self.target) >= 5:
            x, y, z, cls, u = self.target[:5]
            l2 = (f'last target: class {int(cls)}  xyz=({x:.3f}, {y:.3f}, {z:.3f})  uncertainty={u:.2f}', WHITE)
        else:
            l2 = ('last target: none yet', WHITE)
        if self.esp is not None and now - self.t_esp < 2.0:
            e = self.esp
            l3 = (f"ESP32 linked: estop={'YES' if e['estop_latched'] else 'no'}  motors={'ON' if e['motors_enabled'] else 'off'}  "
                  f"cmd_fresh={'yes' if e['cmd_fresh'] else 'no'}  cmds_received={e['cmd_count']}", GREEN)
        else:
            l3 = ('ESP32: no status (agent not running or board not connected - fine for sim-only)', (200, 200, 60))
        l4 = ('keys: e = e-stop, r = reset, q = quit', WHITE)
        return [l1, l2, l3, l4]

    def on_timer(self):
        left = self.spectator if self.spectator is not None else self.raw
        right = self.debug if self.debug is not None else self.raw
        frame = compose_view(left, right, self.status_lines())
        if self.video_path:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
            self.writer.write(frame[:, :, ::-1])
        if self.show:
            cv2.imshow('Automator demo', frame[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                self.estop_pub.publish(Bool(data=True))
                self.get_logger().warn('manual E-STOP sent')
            elif key == ord('r'):
                self.reset_pub.publish(Bool(data=True))
                self.get_logger().info('e-stop reset sent')
            elif key == ord('q'):
                raise KeyboardInterrupt

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()


def main(args=None):
    if not HAS_RCLPY:
        print('[sim_viewer_node] rclpy/cv2 not available; source /opt/ros/humble/setup.bash')
        return
    rclpy.init(args=args)
    node = SimViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
