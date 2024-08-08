from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import is_point_inside_polygon, calculate_center
from shapely.geometry import Point, Polygon
from utils import get_foot_position

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        for i in range(0, len(player_detections)):
            player_detections_first_frame = player_detections[i]
            chosen_player = self.choose_players_inside_bbox(court_keypoints, player_detections_first_frame)

            filtered_player_detections = []
            for player_dict in player_detections:
                filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
                filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players_inside_bbox(self, court_keypoints, player_dict):
        # Create a polygon from court keypoints
        court_polygon = Polygon([(court_keypoints[i], court_keypoints[i+1]) for i in range(0, len(court_keypoints), 2)])

        chosen_players = []
        
        # print(player_dict)
        for track_id, bbox in player_dict.items():
            # Extract bottom left (x_min, y_max) and bottom right (x_max, y_max) points
            # bottom_left = (bbox[0], bbox[3])
            # bottom_right = (bbox[2], bbox[3])
            
            # # Calculate center point
            # center_point = calculate_center(bottom_left, bottom_right)
            center_point = get_foot_position(bbox)

            # Check if the player's bounding box is inside the keypoints zone
            if is_point_inside_polygon(center_point, court_polygon):
                chosen_players.append(track_id)
        
        # If fewer than 5 players are found, return as many as possible
        if len(chosen_players) < 5:
            return chosen_players
        
        # Return the first 5 track IDs
        return chosen_players[:5]

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name=='person':
                player_dict[track_id] = result
            
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding box
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            output_video_frames.append(frame)

        return output_video_frames


