import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
import constants
from utils import (convert_meters_to_pixel_distance, 
                    convert_pixel_distance_to_meters, get_foot_position, 
                    get_closest_keypoint_index, get_height_of_bbox, measure_xy_distance, get_center_of_bbox, measure_distance)

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 150
        self.drawing_rectangle_height = 275
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
    
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
     
    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*60
        
        # point 0 
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        #point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        #point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_WIDTH*2)
        #point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.LONG_SERVICE_LINE_WIDTH)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.LONG_SERVICE_LINE_WIDTH)
        # # #point 11
        # drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[22] = drawing_key_points[18]
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = drawing_key_points[16] 
        drawing_key_points[25] = drawing_key_points[17] + self.convert_meters_to_pixels(constants.MAIN_AREA)
        # # #point 13
        drawing_key_points[26] = drawing_key_points[20] 
        drawing_key_points[27] = drawing_key_points[21] - self.convert_meters_to_pixels(constants.MAIN_AREA)
        # point 14
        drawing_key_points[28] = drawing_key_points[18]
        drawing_key_points[29] = drawing_key_points[19] + self.convert_meters_to_pixels(constants.MAIN_AREA)
        # point 15
        drawing_key_points[30] = drawing_key_points[22]
        drawing_key_points[31] = drawing_key_points[23] - self.convert_meters_to_pixels(constants.MAIN_AREA)
        # point 16
        drawing_key_points[32] = int((drawing_key_points[24] + drawing_key_points[28])/2)
        drawing_key_points[33] = drawing_key_points[29]
        # point 17
        drawing_key_points[34] = int((drawing_key_points[26] + drawing_key_points[30])/2)
        drawing_key_points[35] = drawing_key_points[31]
        # point 18
        drawing_key_points[36] = drawing_key_points[32]
        drawing_key_points[37] = drawing_key_points[19]
        # point 19
        drawing_key_points[38] = drawing_key_points[34] 
        drawing_key_points[39] = drawing_key_points[23]
        # point 20
        drawing_key_points[40] = drawing_key_points[32]
        drawing_key_points[41] = drawing_key_points[1]
        # point 21
        drawing_key_points[42] = drawing_key_points[34] 
        drawing_key_points[43] = drawing_key_points[5]
        # point 22
        drawing_key_points[44] = drawing_key_points[28] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[45] = drawing_key_points[33] 
        # point 23
        drawing_key_points[46] = drawing_key_points[30] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_DISTANCE)
        drawing_key_points[47] = drawing_key_points[31] 
        # point 24
        drawing_key_points[48] = drawing_key_points[2]
        drawing_key_points[49] = drawing_key_points[19] 
        # point 25
        drawing_key_points[50] = drawing_key_points[6] 
        drawing_key_points[51] = drawing_key_points[23] 
        # point 26
        drawing_key_points[52] = drawing_key_points[4] 
        drawing_key_points[53] = drawing_key_points[25]        
        # point 27
        drawing_key_points[54] = drawing_key_points[4] 
        drawing_key_points[55] = drawing_key_points[27]        
        # point 28
        drawing_key_points[56] = drawing_key_points[4] 
        drawing_key_points[57] = drawing_key_points[17]        
        # point 29
        drawing_key_points[58] = drawing_key_points[4] 
        drawing_key_points[59] = drawing_key_points[21]        

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0,1),
            (0,2),
            (3,1),
            (3,2),
            (4,5),
            (7,6),
            (20,16),
            (21,17),
            (27,23),
            (26,22),
            (28,24),
            (29,25)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y), 5, (255,0,0), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame
        
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (225, 225, 225), -1)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1-alpha, 0)[mask]
        return out
    
    def draw_mini_court(self, frames):

        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            output_frames.append(frame)
        
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self, 
                                    object_position, 
                                    closest_keypoint, 
                                    closest_keypoint_index, 
                                    player_height_in_pixels,
                                    player_height_in_meters):

        # get the distance player and keypoint
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_keypoint)

        # convert pixel distance to meters 
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels, 
                                                                            player_height_in_meters,
                                                                            player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels, 
                                                                            player_height_in_meters,
                                                                            player_height_in_pixels)

        # convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (
            self.drawing_key_points[closest_keypoint_index*2],
            self.drawing_key_points[closest_keypoint_index*2+1] 
        )

        mini_court_player_position = (closest_mini_court_keypoint[0]+mini_court_x_distance_pixels,
                                    closest_mini_court_keypoint[1]+mini_court_y_distance_pixels)

        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
            1:constants.PLAYER_1_HEIGHT,
            2:constants.PLAYER_2_HEIGHT,
            11:constants.PLAYER_11_HEIGHT,
            12:constants.PLAYER_12_HEIGHT,
        }

        output_player_boxes= []
        output_ball_boxes= []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [1,2])#1,2
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-2)
                frame_index_max = min(len(player_boxes), frame_num+2)
                # print(player_id)
                # print(player_boxes[i])
                try:
                    bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min,frame_index_max)]
                except KeyError:
                    pass
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [1,2])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for track_id, position in positions[frame_num].items():
                if track_id in [2, 11]:
                    x,y = position
                    x = int(x) - 40
                    y = int(y) - 240
                    cv2.circle(frame, (x,y), 5, (255,255,255), -1)
                else:
                    x,y = position
                    x = int(x) - 20
                    y = int(y) - 60
                    cv2.circle(frame, (x,y), 5, color, -1)
        return frames        

    def draw_shuttle_points_on_mini_court(self, frames, positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for track_id, position in positions[frame_num].items():
                x,y = position
                x = int(x) 
                y = int(y) 
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames  

