import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class FruitDetectorNode(Node):
    def __init__(self):
        super().__init__('fruit_detector_node')
        self.bridge = CvBridge()

        # Subscribe to the aligned color and depth image topics
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',  # Use depth rectified image
            self.depth_callback,
            10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',  # Camera info for depth camera
            self.camera_info_callback,
            10
        )

        self.detection_publisher = self.create_publisher(
            Image,
            'detected_image',
            10
        )

        self.depth_image = None
        self.intrinsics = None

        try:
            self.model = YOLO('weights/strawberry/weights_strawberry_v2.pt')
            self.model.eval()
            self.get_logger().info("YOLOv11 (Fast) model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Error loading YOLOv11 (Fast) model: {e}")

        self.get_logger().info("Fruit Detector Node Initialized!")

    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format with 16-bit encoding
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            if self.depth_image is not None:
                self.get_logger().info(f"Received depth image with shape: {self.depth_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Could not convert depth image: {e}")

    def camera_info_callback(self, msg):
        # Save camera intrinsics (focal lengths, principal points)
        self.intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }
        self.get_logger().info(f"Camera intrinsics: {self.intrinsics}")

    def image_callback(self, msg):
        CLASS_NAMES = ["strawberry", "other"]
        try:
            # Convert color image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # Detect objects using YOLO
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detected_cv_image = cv_image.copy()

        try:
            results = self.model(rgb_image)

            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                print(f"Detected {len(boxes)} objects")

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    conf = confidences[i]
                    cls_id = class_ids[i]
                    class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"Class {cls_id}"

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    label = f"{class_name}: {conf:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(detected_cv_image, (x1, y1), (x2, y2), color, 2)

                    if self.depth_image is not None and self.intrinsics is not None:
                        try:
                            # Get depth value at the detected object's center pixel
                            depth_value = float(self.depth_image[center_y, center_x])  # depth in mm
                            if depth_value == 0 or np.isnan(depth_value) or depth_value < 500 or depth_value > 5000:
                                # Skip invalid or out-of-range depth values
                                print(f"{class_name}: Invalid or out-of-range depth at ({center_x}, {center_y})")
                                label += " | No depth info"
                            else:
                                # Convert mm to meters
                                depth_value /= 1000.0

                                fx = self.intrinsics['fx']
                                fy = self.intrinsics['fy']
                                cx = self.intrinsics['cx']
                                cy = self.intrinsics['cy']

                                # Calculate the 3D position (X, Y, Z)
                                X = (center_x - cx) * depth_value / fx
                                Y = (center_y - cy) * depth_value / fy
                                Z = depth_value

                                # Update the label with 3D coordinates
                                label += f" | X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m"
                                print(f"{class_name} 3D position: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")

                        except Exception as e:
                            self.get_logger().warn(f"Depth or intrinsics error: {e}")
                            label += " | No depth info"
                    else:
                        label += " | No depth info"

                    # Place the label on the bounding box
                    cv2.putText(detected_cv_image, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                print("No objects detected.")

            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.detection_publisher.publish(detection_msg)
            except Exception as e:
                self.get_logger().error(f"Could not convert detected image to ROS Image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
