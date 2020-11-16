### Enviornmnet
Jetson AGX Xavier(Ubuntu 18.04.5)


# Setting

1. ```$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'```

2. ```$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654```

3. ```$ sudo apt update```


# Install

1. ```$ sudo apt install ros-melodic-desktop-full```

2. ```$ sudo apt install python-rosdep```

3. ```$ sudo rosdep init```

4. ```$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc```

5. ```$ source ~/.bashrc```

6. ```$ sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential``` //Install Ros package building tool

# Workspace Init

1. ```$ mkdir -p catkin_ws/src```

2. ```$ cd catkin_ws/

3.```$ catkin_make```

※ if camke error occur, you excute ```unlink /home/parkjiil/catkin_ws/CMakeLists.txt``` // Resolve Link Disengagement to Prevent Crash with Cmake

# Test

-  ```$ roscore```<a href = 'http://wiki.ros.org/roscore'>roscore description</a>

- ```rqt``` <a href = 'http://wiki.ros.org/rqt'>rqt description</a>

- ```rviz``` <a href = 'http://wiki.ros.org/rviz'>rviz description</a>
