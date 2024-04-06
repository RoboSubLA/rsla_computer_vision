import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter("weights", "yolov7.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "cpu", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))
        self.declare_parameter("use_RGB", True, ParameterDescriptor(description="Use webcam"))
        self.declare_parameter("use_depth", False, ParameterDescriptor(description="Use depth camera"))

        self.weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.use_RGB = self.get_parameter("use_RGB").get_parameter_value().bool_value
        self.use_depth = self.get_parameter("use_depth").get_parameter_value().bool_value

        # Camera info and frames
        self.rgb_image = None

        # Flags
        self.camera_RGB = False

        # Timer callback
        self.frequency = 20  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        self.pub_person = self.create_publisher(Point, "/person", 10)
        self.person = Point()

        # Realsense package
        self.bridge = CvBridge()
        
        # Create a video capture object for webcam
        self.cap = cv2.VideoCapture(0)  # Use the first webcam, change index if needed

        # Initialize YOLOv7
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names
                      ]

    def YOLOv7_detect(self):
        """ Perform object detection with YOLOv7 using webcam feed """
        ret, frame = self.cap.read()  # Read frame from webcam
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam")
            return

        # webcam frame processing
        im0 = frame.copy()
        img = frame[np.newaxis, :, :, :]
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    if conf > 0.5: # Limit confidence threshold to 50% for all classes
                        # Draw a boundary box around each object
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        if self.use_depth == True:
                            plot_one_box(xyxy, self.depth_color_map, label=label, color=self.colors[int(cls)], line_thickness=2)

                            label_name = f'{self.names[int(cls)]}'
    
                            # Get box top left & bottom right coordinates
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x = int((c2[0]+c1[0])/2)
                            y = int((c2[1]+c1[1])/2)
    
                            # Limit location and distance of object to 480x680 and 5meters away
                            if x < 480 and y < 640 and self.depth[x][y] < 5000:
                                # Get depth using x,y coordinates value in the depth matrix
                                if self.intr:
                                    real_coords = rs.rs2_deproject_pixel_to_point(self.intr, [x, y], self.depth[x][y])

                                if real_coords != [0.0,0.0,0.0]:
                                    depth_scale = 0.001
                                    # Choose label for publishing position Relative to camera frame
                                    if label_name == 'person':
                                        self.person.x = real_coords[0]*depth_scale
                                        self.person.y = real_coords[1]*depth_scale
                                        self.person.z = real_coords[2]*depth_scale # Depth
                                        self.pub_person.publish(self.person)
                                    if label_name == 'door':
                                        self.door.x = real_coords[0]*depth_scale
                                        self.door.y = real_coords[1]*depth_scale
                                        self.door.z = real_coords[2]*depth_scale # Depth
                                        self.pub_door.publish(self.door)
                                    if label_name == 'stairs':
                                        self.stairs.x = real_coords[0]*depth_scale
                                        self.stairs.y = real_coords[1]*depth_scale
                                        self.stairs.z = real_coords[2]*depth_scale # Depth
                                        self.pub_stairs.publish(self.stairs)
                                    self.get_logger().info(f"depth_coord = {real_coords[0]*depth_scale}  {real_coords[1]*depth_scale}  {real_coords[2]*depth_scale}")

            cv2.imshow("YOLOv7 Object detection result RGB", cv2.resize(im0, None, fx=1.5, fy=1.5))
            if self.use_depth == True:
                cv2.imshow("YOLOv7 Object detection result Depth", cv2.resize(self.depth_color_map, None, fx=1.5, fy=1.5))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def timer_callback(self):
        self.YOLOv7_detect()

    def __del__(self):
        # Release the video capture object when the node is destroyed
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
