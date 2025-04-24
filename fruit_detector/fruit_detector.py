import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# import torch

class FruitDetectorNode(Node):
    def __init__(self):
        super().__init__('fruit_detector_node')
        # Initialize CvBridge to convert ROS Image to OpenCV Image
        self.bridge = CvBridge()
        # Create a subscriber to the RGB image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        # Load your YOLO model (assuming you have a YOLOv5 model)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model
        # self.get_logger().info("Fruit Detector Node Initialized!")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # # Run YOLO model on the image
        # results = self.model(cv_image)  # Runs the YOLOv5 model on the image

        # # Display the results
        # results.show()  # Displays the image with bounding boxes on detected objects

        # # You can also process the results (e.g., publishing, logging, etc.)
        # self.get_logger().info(f"Detected {len(results.xywh[0])} objects")

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
