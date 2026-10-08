"""ROS2 node that plays the physical world: runs the PyBullet sorting scene headless and talks ROS.

Publishes  : /sim/spectator    sensor_msgs/Image rgb8, wide view for humans (only rendered while someone subscribes)
             /camera/image_raw  sensor_msgs/Image rgb8 (rendered from the fixed sim camera, default 3 Hz)
             /camera/reference  sensor_msgs/Image rgb8, latched: the empty scene (background model)
Subscribes : /joint_commands    std_msgs/Float32MultiArray [q1..q4, gripper] -> drives the simulated arm
             /estop             std_msgs/Bool                -> freezes the arm at its current pose

Parameters : objects (int, default 3), seed (int), respawn (bool: refill the table once it is empty),
             inject_unknown (bool: put one random unseen mesh on the table to exercise the OOD e-stop),
             gui (bool: open PyBullet's 3-D window, needs a display), spectator (bool), camera_hz (float)
Runs on the Jetson (PyBullet DIRECT mode, CPU rendering) or any PC on the same ROS_DOMAIN_ID.
"""
import sys
import threading

import numpy as np

from ._paths import find_project_root

sys.path.insert(0, find_project_root())

try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32MultiArray
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object


def spectator_frame(client, w=640, h=480):
    """Wide, human-friendly view of the whole workspace (the perception camera sees only the pick area)."""
    import pybullet as p
    view = p.computeViewMatrix([0.85, 0.75, 0.55], [0.15, 0.12, 0.12], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(50, w / h, 0.05, 3.0)
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)[2]
    return np.reshape(img, (h, w, 4))[:, :, :3].astype(np.uint8)


def rgb_to_msg(rgb, stamp, frame_id='camera'):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = 'rgb8'
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg


class SimBridgeNode(Node):

    def __init__(self):
        super().__init__('sim_bridge_node')
        import pybullet as p
        from sim.scene import NUM_CLASSES, SortScene
        from sim.sim_arm import CTRL_HZ, HOME_Q, PHYS_HZ, SimArm
        self.p, self.NUM_CLASSES, self.HOME_Q = p, NUM_CLASSES, HOME_Q

        self.declare_parameter('objects', 3)
        self.declare_parameter('seed', 0)
        self.declare_parameter('respawn', True)
        self.declare_parameter('inject_unknown', False)
        self.declare_parameter('camera_hz', 3.0)
        self.declare_parameter('gui', False)            # open PyBullet's own 3-D window (needs a display)
        self.declare_parameter('spectator', True)       # also publish /sim/spectator, a nicer view for people

        self.lock = threading.Lock()  # PyBullet is not thread-safe
        gui = bool(self.get_parameter('gui').value)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(1.0, 40, -35, [0.15, 0.12, 0.1], physicsClientId=self.client)
        p.setTimeStep(1.0 / PHYS_HZ, physicsClientId=self.client)
        self.scene = SortScene(self.client)
        self.scene.build_static()
        self.arm = SimArm(self.client)
        self.rng = np.random.default_rng(self.get_parameter('seed').value)
        self.setpoint = np.concatenate([HOME_Q, [1.0]])
        self.estop = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cam_pub = self.create_publisher(Image, '/camera/image_raw', 1)
        self.ref_pub = self.create_publisher(Image, '/camera/reference', latched)
        self.spec_pub = self.create_publisher(Image, '/sim/spectator', 1)
        self.frame_count = 0
        self.create_subscription(Float32MultiArray, '/joint_commands', self.on_cmd, 10)
        self.create_subscription(Bool, '/estop', self.on_estop, latched)

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.client)
        self.reference = self.scene.camera.render(self.client)[0]
        self.ref_pub.publish(rgb_to_msg(self.reference, self.get_clock().now().to_msg()))
        self.spawn_objects()

        self.physics_group = MutuallyExclusiveCallbackGroup()
        self.camera_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / CTRL_HZ, self.on_physics, callback_group=self.physics_group)
        self.create_timer(1.0 / self.get_parameter('camera_hz').value, self.on_camera, callback_group=self.camera_group)
        self.get_logger().info('sim_bridge_node up: PyBullet scene running, publishing /camera/image_raw')

    def spawn_objects(self):
        n = self.get_parameter('objects').value
        classes = [int(c) for c in self.rng.integers(0, self.NUM_CLASSES, n)]
        if self.get_parameter('inject_unknown').value:
            classes[-1] = None  # an unseen random mesh
        with self.lock:
            self.scene.clear_objects()
            self.scene.spawn(classes, self.rng)
        self.get_logger().info(f'Spawned {n} objects')

    def on_cmd(self, msg):
        if len(msg.data) >= 5:
            self.setpoint = np.array(msg.data[:5], dtype=float)

    def on_estop(self, msg):
        self.estop = bool(msg.data)

    def on_physics(self):
        with self.lock:
            if self.estop:  # hold the pose the arm was in when the e-stop fired
                self.arm.command(np.concatenate([self.arm.joint_positions(), [self.setpoint[4]]]), self.scene.objects)
            else:
                self.arm.command(self.setpoint, self.scene.objects)

    def on_camera(self):
        with self.lock:
            rgb, _ = self.scene.camera.render(self.client)
            remaining = sum(1 for pos in self.scene.object_positions()
                            if np.hypot(pos[0], pos[1]) > 0.28 and abs(np.arctan2(pos[1], pos[0])) < 0.8)
        self.cam_pub.publish(rgb_to_msg(rgb, self.get_clock().now().to_msg()))
        self.frame_count += 1
        if self.get_parameter('spectator').value and self.spec_pub.get_subscription_count() > 0:
            with self.lock:
                view = spectator_frame(self.client)
            self.spec_pub.publish(rgb_to_msg(view, self.get_clock().now().to_msg()))
        if remaining == 0 and self.get_parameter('respawn').value and not self.arm.held:
            self.spawn_objects()


def main(args=None):
    if not HAS_RCLPY:
        print('[sim_bridge_node] rclpy not available; source /opt/ros/humble/setup.bash')
        return
    rclpy.init(args=args)
    node = SimBridgeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
