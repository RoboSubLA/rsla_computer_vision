import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, \
    ObjectHypothesisWithPose, Pose2D
from cv_bridge import CvBridge
from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch
import cv2
import numpy as np
from typing import List, Union


def get_random_color(seed):
    gen = np.random.default_rng(seed)
    color = tuple(gen.choice(range(256), size=3))
    color = tuple([int(c) for c in color])
    return color

def draw_detections(img: np.array, bboxes: List[List[int]], classes: List[int],
                    class_labels: Union[List[str], None]) -> np.array:
    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        color = get_random_color(int(cls))
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3
        )
    
        if class_labels:
            label = class_labels[int(cls)]

            x_text = int(x1)
            y_text = max(15, int(y1 - 10))
            img = cv2.putText(
                img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA
            )

    return img

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("yolov7_node") 

        self.declare_parameter("weights", "best.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "cuda", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))
        self.declare_parameter("visualize", True, ParameterDescriptor(description="Visualize detections"))

        self.weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.visualize = self.get_parameter("visualize").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.model = attempt_load(self.weights, map_location=self.device).eval()
        self.publisher = self.create_publisher(Detection2DArray, "yolov7_detections", 10)
        if self.visualize:
            self.visualization_publisher = self.create_publisher(Image, "yolov7_detections/visualization", 10)

        self.capture = cv2.VideoCapture(0)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        h, w, _ = frame.shape
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        return img_tensor

    def _detect_objects(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        detections = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]
        return detections

    def _create_detection_msg(self, img_msg: Image, detections: torch.Tensor) -> Detection2DArray:
        detection_array_msg = Detection2DArray()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.tolist()
            single_detection_msg = Detection2D()

            # Convert the raw image data to a ROS Image message
            img_ros_msg = self.bridge.cv2_to_imgmsg(img_msg, "bgr8")
            single_detection_msg.source_img = img_ros_msg

            # bbox
            bbox = BoundingBox2D()
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            cx = int(round(x1 + w / 2))
            cy = int(round(y1 + h / 2))
            bbox.size_x = w
            bbox.size_y = h

            bbox.center = Pose2D()
            bbox.center.x = cx
            bbox.center.y = cy

            single_detection_msg.bbox = bbox

            # class id & confidence
            obj_hyp = ObjectHypothesisWithPose()
            obj_hyp.id = int(cls)
            obj_hyp.score = conf
            single_detection_msg.results = [obj_hyp]

            detection_array_msg.detections.append(single_detection_msg)

        return detection_array_msg

    def timer_callback(self):
        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam")
            return

        try:
            # Preprocess the image for YOLOv7
            img_tensor = self._preprocess_image(frame)
            detections = self._detect_objects(img_tensor)
            detection_msg = self._create_detection_msg(frame, detections)
            self.publisher.publish(detection_msg)

            if self.visualize:
                bboxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in detections[:, :4].tolist()]
                classes = [int(cls) for cls in detections[:, 5].tolist()]
                vis_img = draw_detections(frame, bboxes, classes, None)
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
                self.visualization_publisher.publish(vis_msg)

            # Display the webcam feed
            cv2.imshow('Webcam Feed', frame)
            cv2.waitKey(1)  # Wait for a short time to refresh the display

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def __del__(self):
        if self.capture.isOpened():
            self.capture.release()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
