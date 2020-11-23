# ROS MASTER

## Exctue ROS Master

- Excute command ```$ roscore```

- Termination command ```ctrl+c```

<img src='https://user-images.githubusercontent.com/47775179/99933767-51c75500-2d9f-11eb-8e27-90dda19477ae.png'></img>

## Excute Lidar Node

- ```$ cd $TRAM_HOME/tram_lidar/config```

- ```$ gedit $TRAM_HOME/tram_lidar/config/config.ini```

- Lidar sensor node setup file

<img src='https://user-images.githubusercontent.com/47775179/99933861-93f09680-2d9f-11eb-97ae-461ae855de0a.png'></img>

- Excute Node ```$roslaunch tram lidar_nodes_start.launch```

- If you excute lidar sensor node, the topic will be published(
 
  - (Topic Name[msg type])

    - Distance Information : /parsed_tx/object_data_2221 [ibeo_msgs/ObjectData2221]
    
    - Scan Information :  /parsed_tx/scan_data_2202 [ibeo_msgs/ScanData2202]
    
    - Object Visual Information : /lidar_MarkerArray_objects [visualization_msgs/MarkerArray]
    
    - Point Information : /as_tx/point_cloud [sensor_msgs/PointCloud2]
    
    - Adjusted Point Information :  /Calib_lidar_MarkerArray_objects [visualization_msgs/MarkerArray]
    
    - Shape Information :  /as_tx/object_contour_points [visualization_msgs/Marker]
    
    - Object Visual Information : /as_tx/objects_ibeo [visualization_msgs/MarkerArray]
    
    - Error/Warning Information :  /parsed_tx/error_warning [ibeo_msgs/ErrorWarning]
    
    - Vehicle Status Information :  /parsed_tx/host_vehicle_state_2805 [ibeo_msgs/HostVehicleState2805]



