from __future__ import print_function

import argparse
import cv2
import rosbag
from cv_bridge import CvBridge


IMAGE_TOPIC_TYPE = 'sensor_msgs/Image'
IMAGE_COLOR_FORMAT = 'rgb8'


class ImageTopic(object):
    ''' Contains information for a ROS image topic. '''
    def __init__(self, name, msgs, fps):
        self.name = name
        self.fps = fps
        self.msgs = msgs

        # Convert Image messages to cv2 Image types.
        bridge = CvBridge()
        self.images = [bridge.imgmsg_to_cv2(msg, IMAGE_COLOR_FORMAT)
                       for msg in msgs]

        # We need to be careful here, as numpy gives us (rows, cols) but the
        # OpenCV frame size expects (cols, rows).
        rows, cols, _ = self.images[0].shape
        self.frame_size = (cols, rows)


def topic_name_to_file_name(topic_name, ext):
    ''' Convert a ROS topic name to a file name. '''
    # Replace leading slash, replace any other slashes with underscores, and
    # append the file extension.
    if topic_name[0] == '/':
        topic_name = topic_name[1:]
    return topic_name.replace('/', '_') + '.' + ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bagfile', help='Bagfile to parse Image streams from.')
    parser.add_argument('-e', '--encoding', choices=['avi'], default='avi',
                        help='Video encoding of output file.')
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    info = bag.get_type_and_topic_info()
    topics = info[1]
    image_topics = []

    # Create a list of all image topics from the bag.
    for topic, topic_info in topics.iteritems():
        topic_type = topic_info[0]
        fps = topic_info[3]

        if topic_type == IMAGE_TOPIC_TYPE:
            msgs = [msg for _, msg, _ in bag.read_messages(topic)]
            image_topics.append(ImageTopic(topic, msgs, fps))

    # Convert the images from each topic into a video.
    for topic in image_topics:
        file_name = topic_name_to_file_name(topic.name, 'avi')
        print('Writing topic {} to {}.'.format(topic.name, file_name))
        vid_writer = cv2.VideoWriter(file_name, fourcc, topic.fps,
                                     topic.frame_size)
        for image in topic.images:
            vid_writer.write(image)
        vid_writer.release()

    bag.close()


if __name__ == '__main__':
    main()
