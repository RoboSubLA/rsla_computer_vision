## Getting started 
How to Train YOLOv7 on a Custom Dataset using Google Colab
## Step 1: Creating project
Before you start, you need to create a Roboflow account. Once you do that, you can create a new project in the Roboflow dashboard. Keep in mind to choose the right project type. In our case, Object Detection.
![](https://camo.githubusercontent.com/7264a275db6473f8432239dce3b8e1819208b5421d40654afce57680d6fd3311/68747470733a2f2f6d656469612e726f626f666c6f772e636f6d2f707265706172696e672d637573746f6d2d646174617365742d6578616d706c652f6372656174696e672d70726f6a6563742e6769663f696b2d73646b2d76657273696f6e3d6a6176617363726970742d312e342e33267570646174656441743d31363732393239373939383532)

## Step 2: Uploading images
Next, add the data to your newly created project. You can do it via API or through our web interface.
If you drag and drop a directory with a dataset in a supported format, the Roboflow dashboard will automatically read the images and annotations together.

## Step 3: Labeling
If you only have images, you can label them in Roboflow Annotate.
For best accuracy, annotate the image as close to its borders as possible, otherwise, the model will train on background pixels.
![](https://i.ibb.co/m5Fj8kH/Screenshot-2024-03-11-151736.jpg)

## Step 4: Generate new dataset version
Now that we have our images and annotations added, we can Generate a Dataset Version. When Generating a Version, you may elect to add preprocessing and augmentations. This step is completely optional, however, it can allow you to significantly improve the robustness of your model.


## Step 5: Exporting dataset
Once the dataset version is generated, we have a hosted dataset we can load directly into our notebook for easy training. Click Export and select the YOLO v5 PyTorch dataset format.
![](https://camo.githubusercontent.com/bda73b306551a545469d21838128066bb42d3e9cf403b756aac2620ed90a840c/68747470733a2f2f6d656469612e726f626f666c6f772e636f6d2f707265706172696e672d637573746f6d2d646174617365742d6578616d706c652f6578706f72742e6769663f696b2d73646b2d76657273696f6e3d6a6176617363726970742d312e342e33267570646174656441743d31363732393433333133373039)
sources: https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov7-object-detection-on-custom-data.ipynb

## Using TrainYOLOV7 in Google Colab
download file and open on YOUR google colab account
[HERE](https://store.augmentedstartups.com/605618c8-d8c8-4a25-be71-852308115520)

## Running YOLOV7 on custom trained dataset 
There are many ways to run but using google colab is the simplest.
You can run the same command on your UBUNTU terminal, assuming you have CUDA, PYTHON3, your weights, and test images installed on your local device

### BEFORE YOU RUN THIS CODE
make sure you are using the correct path from YOUR trained model and testing on your uploaded images from YOUR path in your google drive \
```!python detect.py --weights /content/gdrive/MyDrive/yolov7/runs/train/exp2/weights/best.pt --conf 0.1 --source /content/gdrive/MyDrive/yolov7/Trash-5/test/images```


## clone yolov7 
There may be other ways to do it but download the repo from [yolo](https://github.com/WongKinYiu/yolov7) \
assuming you have github desktop version and have your github linked to this repo and [GIT](https://git-scm.com/downloads) installed

copy paste all files from "yolov7" folder into a new folder (for organizational purposes) your github directory\
cd into your github directory\
```git add .``` \
```git commit -m "[insert your commit comment]"``` \
```git push origin``` \
if you need help [here](https://youtu.be/ueQs5pQ8ZMM)

## ROS2 Installation 
`locale  # check for UTF-8` \
`sudo apt update && sudo apt install locales`\
`sudo locale-gen en_US en_US.UTF-8` \
`sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8` \
`export LANG=en_US.UTF-8`\
`Locale  # verify settings`

```sudo apt install software-properties-common``` \
```sudo add-apt-repository universe``` \
```sudo apt update && sudo apt install curl -y ``` \
```sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg``` 

```echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null```

```sudo apt update``` \
```sudo apt upgrade``` \
Desktop Install for ROS, RViz, demos, tutorials \
```sudo apt install ros-iron-desktop```

### Environment setup
Replace ".bash" with your shell if you're not using bash\
Possible values are: setup.bash, setup.sh, setup.zsh\
```source /opt/ros/iron/setup.bash ```

### Watch ROS2 tutorial videos for futher help
[YOUTUBE](https://www.youtube.com/@RoboticsBackEnd)


## CV ROS2 Node
This is an example to work on your LOCAL computer\
to clone and update the github will be a slightly different processs\
`mkdir ros2_ws`\
`cd ros2_ws`\
`mkdir src`\
`colcon build`\
`source ~/ros2_ws/install/setup.bash`

### Creating a package
`cd ros2_ws` \
`cd src` \
`ros2 pkg create wailord_cv --build-type ament_python --dependencies rclpy` \
`cd ..` \
`colcon build` \

python packages \
`pip3 list` \
`pip3 list | grep setuptools` \
downgrade to 58.0.2 if things arent working correctly \
`pip3 install setuptools==58.2.0`
run `colcon build` again

`cd ..` \
`cd src` \
`cd wailord_cv` \
`cd wailord_cv` #yes twice \
`touch my_first_node.py`\
make it runnable `chmod +x my_first_node.py`

update node in vscode\
```  #!/usr/bin/python3
import rclpy
from rclpy.node import Node

class MyNode(Node):

    def __init__(self):
        super().__init__("first_node")
        self.get_logger().info("Hello from ROS2")

def main(args=None):
    #initialize ros2 features
    rclpy.init(args=args)
    
    #initialize node
    node = MyNode()
    
    
    rclpy.shutdown()


if __name__== '__main__':
    main() 
```

cd into the directory and run node 
`./my_first_node.py` 

### update setup.py
```
entry_points={
        'console_scripts': [
            "test_node = wailord_cv.my_first_node:main"
        ],
    }
```
Go to workspace directory \
`colcon build` 

you need to use your path and edit it in bashrc so your ros packages work on every terminal \
and you dont have to source your setup.bash in everytime you open a new terminal \
`source ~/install/setup.bash` \
`ros2 run wailord_cv test_node`

CONGRATULATION YOU MADE YOUR FIRST NODE \
commands for making new python files and making them executable, example py file name "pose_subscriber" \
`touch pose_subscriber.py` \
`chmod +x pose_subscriber.py`

PACKAGES NEEDED \
`pip install torch torchvision torchaudio`\
`pip install opencv-python-headless`



important commands for subscriber \
`ros2 topic list` for looking up commands \
`ros2 topic info ______` insert type, for looking up type \
`ros2 interface show _____` insert type, for looking up cmds

