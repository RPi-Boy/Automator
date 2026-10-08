"""ROS2 perception node: camera frame -> detections -> shape class, refined xyz, OOD uncertainty.

Subscribes : /camera/image_raw   sensor_msgs/Image (rgb8 or bgr8)
             /camera/reference   sensor_msgs/Image, latched: the same view with no objects (background model)
             /arm_busy           std_msgs/Bool from joint_trajectory_node: perception pauses while the arm moves
Publishes  : /perception/debug  sensor_msgs/Image: the frame with boxes, class and uncertainty (only built while subscribed)
             /arm_target         std_msgs/Float32MultiArray [x, y, z, class_id, uncertainty]
             - the least-uncertain known object, or
             - if ANY object in the pick area is out-of-distribution (uncertainty > threshold), that object,
               so the downstream e-stop logic fires.

Model: .engine (TensorRT, build on the Jetson), .onnx (ONNX Runtime) or .pt - see perception/inference.py.
"""
import os
import sys
import time

import numpy as np

from ._paths import find_project_root  # noqa: E402
from .overlay import draw_detections  # noqa: E402

sys.path.insert(0, find_project_root())

from perception.camera import SIM_CAMERA, PinholeCamera  # noqa: E402
from perception.detector import detect, in_pick_region  # noqa: E402
from perception.inference import load_backend  # noqa: E402
from perception.targeting import choose_target  # noqa: E402

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32MultiArray
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object


def image_to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB, honouring row stride and encoding (no cv_bridge needed)."""
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = row[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1].copy() if msg.encoding == 'bgr8' else img.copy()


def perceive(backend, camera, rgb, reference, with_boxes=False):
    """Detect objects in the pick area and run the network on all crops in one batch.
    Returns [{'xyz', 'cls', 'unc'}] (xyz = geometric back-projection + learned residual)."""
    dets = [d for d in detect(rgb, reference, camera) if in_pick_region(d['geom_xyz'][:2])]
    if not dets:
        return []
    crops = np.stack([d['crop'] for d in dets]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    bbox = np.stack([d['bbox_feat'] for d in dets]).astype(np.float32)
    logits, offset, unc = backend(np.ascontiguousarray(crops), bbox)
    return [{'xyz': d['geom_xyz'] + offset[i], 'cls': int(logits[i].argmax()), 'unc': float(unc[i]), 'bbox': d['bbox']}
            for i, d in enumerate(dets)]


class EvidentialVisionNode(Node):

    def __init__(self):
        super().__init__('evidential_vision_node')
        root = find_project_root()
        self.declare_parameter('model_path', os.path.join(root, 'models', 'perception.onnx'))
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('uncertainty_threshold', 0.75)
        self.declare_parameter('settle_s', 0.6)         # wait this long after the arm stops before looking
        self.declare_parameter('reference_image', '')   # optional PNG instead of the /camera/reference topic

        self.backend = load_backend(self.get_parameter('model_path').value)
        self.camera = PinholeCamera(**SIM_CAMERA)
        self.threshold = self.get_parameter('uncertainty_threshold').value
        self.min_period = 1.0 / self.get_parameter('rate_hz').value
        self.reference = None
        self.last_run = 0.0
        self.settle = self.get_parameter('settle_s').value
        self.arm_busy = False
        self.idle_since = 0.0

        ref_path = self.get_parameter('reference_image').value
        if ref_path:
            import cv2
            self.reference = cv2.imread(ref_path)[:, :, ::-1].copy()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, '/camera/reference', self.on_reference, latched)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 1)
        self.create_subscription(Bool, '/arm_busy', self.on_busy, 1)
        self.pub = self.create_publisher(Float32MultiArray, '/arm_target', 10)
        self.dbg_pub = self.create_publisher(Image, '/perception/debug', 1)   # annotated frame, only built when watched
        self.get_logger().info(f'Vision node up: backend={self.backend.name}, '
                               f'reference={"loaded" if self.reference is not None else "waiting"}')

    def on_reference(self, msg):
        self.reference = image_to_rgb(msg)
        self.get_logger().info('Background reference received')

    def on_busy(self, msg):
        if self.arm_busy and not msg.data:
            self.idle_since = time.monotonic()
        self.arm_busy = bool(msg.data)

    def on_image(self, msg):
        now = time.monotonic()
        # While the arm moves, its own links, gripper and the carried object cross the pick area and are not in
        # the background reference, so they would be flagged as unknown objects. Look only when it is idle and
        # settled. Trade-off: an unknown object entering the workspace mid-motion is caught only once the arm stops.
        if self.arm_busy or now - self.idle_since < self.settle:
            return
        if self.reference is None or now - self.last_run < self.min_period:
            return
        self.last_run = now
        t0 = time.perf_counter()
        rgb = image_to_rgb(msg)
        results = perceive(self.backend, self.camera, rgb, self.reference)
        ms = (time.perf_counter() - t0) * 1000
        if self.dbg_pub.get_subscription_count() > 0:
            dbg = draw_detections(rgb, results, self.threshold, ms)
            out_img = Image(height=dbg.shape[0], width=dbg.shape[1], encoding='rgb8', step=dbg.shape[1] * 3, data=dbg.tobytes())
            out_img.header = msg.header
            self.dbg_pub.publish(out_img)
        target = choose_target(results, self.threshold)
        if target is None:
            return
        out = Float32MultiArray()
        out.data = [float(v) for v in target['xyz']] + [float(target['cls']), target['unc']]
        self.pub.publish(out)
        self.get_logger().info(
            f"target class {target['cls']} xyz {np.round(target['xyz'], 3).tolist()} "
            f"unc {target['unc']:.3f} ({(time.perf_counter() - t0) * 1000:.1f} ms, {len(results)} objects)",
            throttle_duration_sec=1.0)


def main(args=None):
    if not HAS_RCLPY:
        print('[evidential_vision_node] rclpy not available; source /opt/ros/humble/setup.bash')
        return
    rclpy.init(args=args)
    node = EvidentialVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
