"""ROS2 node: /arm_target -> analytic IK -> quintic joint trajectory -> /joint_commands.

Subscribes : /arm_target  std_msgs/Float32MultiArray [x, y, z, class_id, uncertainty]
             /estop_reset std_msgs/Bool (True clears the latch here; the ESP32 listens too)
Publishes  : /arm_busy       std_msgs/Bool (true while a pick-and-place is executing; the vision node pauses)
             /joint_commands std_msgs/Float32MultiArray [q1..q4 (rad), gripper (-1 closed .. +1 open)] @ rate_hz
             /estop          std_msgs/Bool (latched high when uncertainty > threshold)
"""
from .config import TOOL_LENGTH
from .trajectory import PickPlaceSequencer

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32MultiArray
    from rclpy.qos import QoSProfile, DurabilityPolicy
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object


class JointTrajectoryNode(Node):

    def __init__(self):
        super().__init__('joint_trajectory_node')
        self.declare_parameter('uncertainty_threshold', 0.75)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('place_xyz', [-0.15, 0.30, 0.03])
        # Optional flat [x0, y0, z0, x1, y1, z1, ...] bin positions for class 0, 1, ...; empty = shared defaults
        self.declare_parameter('place_by_class', [0.0])
        self.declare_parameter('tool_length', TOOL_LENGTH)
        thr = self.get_parameter('uncertainty_threshold').value
        self.rate = self.get_parameter('rate_hz').value
        self.seq = PickPlaceSequencer(
            place_xyz=list(self.get_parameter('place_xyz').value),
            place_by_class=self._bins(self.get_parameter('place_by_class').value) or None,
            tool_length=self.get_parameter('tool_length').value,
            uncertainty_threshold=thr)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Float32MultiArray, '/joint_commands', 10)
        self.estop_pub = self.create_publisher(Bool, '/estop', latched)
        self.busy_pub = self.create_publisher(Bool, '/arm_busy', 1)
        self.create_subscription(Float32MultiArray, '/arm_target', self.on_target, 10)
        self.create_subscription(Bool, '/estop_reset', self.on_reset, 10)
        self.create_timer(1.0 / self.rate, self.on_timer)
        self.estop_pub.publish(Bool(data=False))
        self.get_logger().info(f'joint_trajectory_node up: {self.rate:.0f} Hz, e-stop threshold {thr}')

    @staticmethod
    def _bins(flat):
        flat = list(flat)
        return {i: flat[3 * i:3 * i + 3] for i in range(len(flat) // 3)}  # [0.0] (default) -> {}

    def on_target(self, msg):
        if len(msg.data) < 5:
            self.get_logger().warn('arm_target needs 5 values [x, y, z, class, uncertainty]')
            return
        x, y, z, cls, unc = msg.data[:5]
        was_halted = self.seq.halted
        if self.seq.on_target([x, y, z], unc, int(cls)):
            self.get_logger().info(f'Planned pick of ({x:.3f}, {y:.3f}, {z:.3f}), uncertainty {unc:.3f}')
        elif self.seq.halted and not was_halted:
            self.estop_pub.publish(Bool(data=True))
            self.get_logger().error(f'E-STOP: {self.seq.last_reject}')
        elif self.seq.last_reject != 'busy':
            self.get_logger().warn(f'Target rejected: {self.seq.last_reject}')

    def on_reset(self, msg):
        if msg.data:
            self.seq.clear_halt()
            self.estop_pub.publish(Bool(data=False))
            self.get_logger().info('E-stop latch cleared')

    def on_timer(self):
        # Keep streaming the held pose when idle: the ESP32 watchdog cuts the drivers if commands stop.
        out = Float32MultiArray()
        out.data = [float(v) for v in self.seq.tick(1.0 / self.rate)]
        self.cmd_pub.publish(out)
        self.busy_pub.publish(Bool(data=self.seq.busy))


def main(args=None):
    if not HAS_RCLPY:
        print('[joint_trajectory_node] rclpy not available; source /opt/ros/humble/setup.bash')
        return
    rclpy.init(args=args)
    node = JointTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
