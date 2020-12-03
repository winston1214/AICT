## Install Lidar Driver

- <a href='http://wiki.ros.org/ibeo_lux'>wiki</a>

```$ cd ~/catkin_ws/src```

```$ git clone https://github.com/astuff/ibeo_lux.git```

```$ cd ..```

```$ catkin_make ```

## Install Radar Driver

```$ sudo apt install ros-$ROS_DISTRO-kvaser-inference ```

```$ sudo apt install ros-$ROS_DISTRO-delphi-esr```

## Install IMU sensor Driver

- <a href='http://wiki.ros.org/xsens_driver'>wiki</a>

- <a href='http://wiki.ros.org/imu_filter_madgwick'>wiki</a>

```$ cd ~/catkin_ws/src```

```$ git clone https://github.com/ethz-asl/ethzasl_xsens_driver.git```

```$ cd ~/catkin_ws/src```

```$ git clone https://github.com/ccny-ros-pkg/imu_tools.git```

```$ cd ~/catkin_ws```

```$ catkin_make -DCATKIN_WHITELIST_PACKAGES="xsens_driver"```

```$ catkin_make -DCATKIN_WHITELIST_PACKAGES="imu_complementary_filter"```

```$ catkin_make -DCATKIN_WHITELIST_PACKAGES="imu_filter_madgwick"```

# Install Camera Sensor driver

- <a href='http://wiki.ros.org/pylon_camera'>wiki</a>

- First, <a href='https://www.baslerweb.com/de/support/downloads/downloads-software/'>sdk download </a>
  - pylon 5.2.0 Camera Software Suite Linux x86(64 Bit) - Debian Installer Package

- Install SDK

```$ cd ~/Downloads/```

```$ sudo dpkg -i pylon_5.2.0.13457-deb0_amd64.deb```

- Configure ROS Package Dependency

``` $ sudo sh -c 'echo "yaml https://raw.githubusercontent.com/magazino/pylon_camera/indigo-devel/rosdep/pylon_sdk.yaml " > /etc/ros/rosdep/sources.list.d/15-plyon_camera.list'```

```$ rosdep update```

- Package Clone & build

```$  d ~/catkin_ws/src/ && git clone https://github.com/magazino/pylon_camera.git && git clone https://github.com/magazino/camera_control_msgs.git```

```$ rosdep install --from-paths . --ignore-src —rosdistro=$ROS_DISTRO -y```

```$cd ~/catkin_ws```

``` $catkin_make –DCATKIN_WHITELIST_PACKAGES="pylon_camera"```

# Install Tram Sensor Node (Secret)
