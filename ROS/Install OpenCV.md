# How to install opencv if you are upgrading to Python 3.6 in ROS1(The original opencv is pre-installed (in python2.7))

## Environment

- ROS melodic, Ubuntu 18.04, Jetson AGX Xavier


## How to install the latest version of OpenCV

- Installing OpenCV from the Source
```
$ sudo apt install build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
 ```

- git clone opencv & opencv contrib

```$ mkdir ~/opencv && cd ~/opencv```

```$ git clone https://github.com/opencv/opencv.git ```

```$ git clone https://github.com/opencv/opencv_contrib.git```

***default version : Latest opencv ver***

```$ cd ~/opencv/opencv```

- build

```$ mkdir build && cd build```

```
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON .. 
```

- make & install

```$ make -j8 ```

**Modify the -j flag according to your processor. If you do not know the number of cores your processor, you can find it by typing ```nproc```.**

```$ sudo make install ```


## Test
```$ python3```

```>>> import cv2```

```>>> print(cv2.__version__)```
