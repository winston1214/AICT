### Enviornmnet
Jetson AGX Xavier(Ubuntu 18.04.5)

# Visualization

1. ``` $ roscore```

-- Open New terminal --

If you want to do it indefinitely,

2. ``` $ roslaunch playback.launch file:=BagFilePath``` // palyback.launch = Infinite repetition

※ playback.launch file
```
<launch>
  <!-- Creates a command line argument called file -->
  <arg name="file"/>

  <!-- Run the rosbag play as a node with the file argument -->
  <node name="rosbag" pkg="rosbag" type="play" args="--loop $(arg file)" output="screen"/>
</launch>
```

else

2. ```rospaly bagFilepath```

-- Open New terminal --

3. ```rviz```

## Rviz Setting(rosplay BagFile)

- Fixed Frame setting : ibeo_lux

- click "add"

- Choose the visualization option you want.

- I choose PointArray and Image

<img src='https://user-images.githubusercontent.com/47775179/99535790-be7fcf80-29ec-11eb-8eab-fbe661b1d96e.png'>Result</src>





