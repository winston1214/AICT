## Environmnet

- Ubuntu 18.04, ROS melodic1

**When you use python3 in ROS1, you may see cv_bridge error.How to solve this problem?**

# Solution in ROS melodic(python3.6)

- Install package
```
$ sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge
```


- create workspace
```
$ mkdir -p catkin_cvws/src
$ cd catkin_cvws
$ catkin init
```

- Instruct catkin to set cmake variables
```
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

- Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
```
$ catkin config --install
```

- git clone
```
$ git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
```

- Find version of cv_bridge in your repository
```
$ apt-cache show ros-melodic-cv-bridge | grep Version
```
output example : Version: **1.12.8**-0xenial-20180416-143935-0800

```
cd src/vision_opencv/
git checkout 1.12.8
cd ../../
```
- catkin build
```
$ catkin build cv_bridge
```
※ Unfortunately, using ```catkin_make``` causes errors

- Extend environment with new package
```
$ source install/setup.bash --extend
```
- Make it available to all workspaces
```
echo "source ~/catkin_cvws/install/setup.bash" >> ~/.bashrc
```
- Test
```
$ python3
python 3.6.9
>>> from cv_bridge.boost.cv_bridge_boost import getCvType
>>>
```

If there is no error, it is a success.
