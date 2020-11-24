import rosbag
import numpy as np
from geometry_msgs.msg import Point
bag = rosbag.Bag('twenty.bag')
topic = '/parsed_tx/object_data_2221'
frame = 1
with open('twenty.txt','a') as f:
    f.write('frmae,class,distance\n')
    for topic ,msg,t in bag.read_messages(topics=topic):
        for i in range(len(msg.object_list)):
            x = msg.object_list[i].bounding_box_center.x
            y = msg.object_list[i].bounding_box_center.y
            distance = np.sqrt(np.power(x,2) + np.power(y,2))
            classification = msg.object_list[i].classification
            if classification in [1,3,5]:
                f.write('{},{},{}\n'.format(frame,classification,distance))
            else:
                pass
        frame+=1
