from shapely.geometry import Point, Polygon


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)
    
    return center_x, center_y

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def is_point_inside_polygon(point, polygon):
    """
    Check if a point (x, y) is inside a polygon.
    """
    point_obj = Point(point[0], point[1])
    return polygon.contains(point_obj)

def calculate_center(bottom_left, bottom_right):
    """
    Calculate the center point between bottom left and bottom right corners of a bounding box.
    """
    bottom_left_point = Point(bottom_left[0], bottom_left[1])
    bottom_right_point = Point(bottom_right[0], bottom_right[1])
    
    center_x = (bottom_left_point.x + bottom_right_point.x) / 2
    center_y = (bottom_left_point.y + bottom_right_point.y) / 2
    return (center_x, center_y)

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1+x2)/2 , y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    # print(keypoints)
    for keypoint_indix in keypoint_indices:
        # print(keypoint_indix*2)
        # print(keypoint_indix*2+1)
        keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
    
    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))