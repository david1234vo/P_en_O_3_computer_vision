from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import kinect_packages.kinect_working_version_sensor
from kinect_packages.kinect_working_version_sensor import TopDownViewRuntime


def convert_to_proper_coordinate(location):
    window_size = (1920, 1080)
    horizontal_factor = 1 / 1000
    vertical_factor = -1 / 1000
    x, y, depth = location
    width, height = window_size
    horizontal_coordinate = (x - width / 2) * depth * horizontal_factor
    vertical_coordinate = (y - height / 2) * depth * vertical_factor
    return [horizontal_coordinate, vertical_coordinate, depth]


def get_hands_location(bodies=None, kinect=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color
                    | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)):
    """
        returns a list containing tuples which all contain two tuples, respectively storing
        the xyz coordinates of the left hand and the right hand
    """

    hands_locations = []
    if bodies is not None:
        for body in bodies:
            joints = body.joints
            lx = joints[PyKinectV2.JointType_HandLeft].Position.x
            ly = joints[PyKinectV2.JointType_HandLeft].Position.y
            lz = joints[PyKinectV2.JointType_HandLeft].Position.z

            rx = joints[PyKinectV2.JointType_HandRight].Position.x
            ry = joints[PyKinectV2.JointType_HandRight].Position.y
            rz = joints[PyKinectV2.JointType_HandRight].Position.z

            hands_locations.append(((lx, ly, lz), (rx, ry, rz)))

    return hands_locations




def hands_to_close(bodies, distance_allowed):
    """
        prints whether or not the hands of any two people are to close
    """
    while True:
        hand_locations = get_hands_location(bodies)
        if len(hand_locations) > 1:
            for i in range(len(hand_locations) - 1):
                for k in range(0, 2):
                    x1, y1, z1 = convert_to_proper_coordinate(hand_locations[i][k])

                    for j in range(len(hand_locations[i+1:])):
                        for q in range(0, 2):
                            x2, y2, z2 = convert_to_proper_coordinate(hand_locations[j][q])


                            current_distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                            if current_distance < distance_allowed:
                                print("too close: hands at", hand_locations[i][k], "and", hand_locations[j][q])
