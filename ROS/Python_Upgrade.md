# How to Python Upgrade(python2.7 -> python3.6) in Ros

## Environment

Ubuntu 18.04, ROS melodic

## Install Libraries

```
$ sudo apt-get install python3-pip python3-yaml
```
```
$ sudo pip3 install rospkg catkin_pkg
```

- If occur cv_bridge error

Create New catkin_build_ws directory

```
$ mkdir ~/catkin_build_ws && cd ~/catkin_build_ws
```
```
$ catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/bin/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```
```
$ catkin config --install
```
```
$ mkdir src && cd src
```
```
$ git clone -b melodic https://github.com/ros-perception/vision_opencv.git
```
```
$ cd ..
$ catkin build cv_bridge
```
```
$ source install/setup.bash --extend
```


※If the error still occurs, click <a href='https://github.com/winston1214/AICT/blob/master/ROS/python3%20cv_bridge.md'>this</a>
