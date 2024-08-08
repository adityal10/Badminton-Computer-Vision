from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class ShuttleTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_shuttle_positions(self, shuttle_positions):
        shuttle_positions = [x.get(1, []) for x in shuttle_positions]
        # convert list in to pd dataframe
        df_shuttle_positions = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate the missing values
        df_shuttle_positions = df_shuttle_positions.interpolate()
        df_shuttle_positions = df_shuttle_positions.bfill()

        shuttle_positions  = [{1:x} for x in df_shuttle_positions.to_numpy().tolist()]
        
        return shuttle_positions

    def get_ball_shot_frames(self, shuttle_positions):
        shuttle_positions = [x.get(1, []) for x in shuttle_positions]
        # convert list in to pd dataframe
        df_shuttle_positions = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_shuttle_positions['mid_y'] = (df_shuttle_positions['y1'] + df_shuttle_positions['y2'] )/2
        df_shuttle_positions['mid_y_rolling_mean'] = df_shuttle_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_shuttle_positions['delta_y'] = df_shuttle_positions['mid_y_rolling_mean'].diff()

        df_shuttle_positions['ball_hit'] = 0

        minimum_change_frames_for_hit = 7
        for i in range(1, len(df_shuttle_positions)-int(minimum_change_frames_for_hit*1.2)):
            negative_position_change = df_shuttle_positions['delta_y'].iloc[i] > 0 and df_shuttle_positions['delta_y'].iloc[i+1] < 0 
            positive_position_change = df_shuttle_positions['delta_y'].iloc[i] < 0 and df_shuttle_positions['delta_y'].iloc[i+1] > 0 

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_shuttle_positions['delta_y'].iloc[i] > 0 and df_shuttle_positions['delta_y'].iloc[change_frame] < 0 
                    positive_position_change_following_frame = df_shuttle_positions['delta_y'].iloc[i] < 0 and df_shuttle_positions['delta_y'].iloc[change_frame] > 0 
                    
                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
                
                if change_count>minimum_change_frames_for_hit-1:
                    df_shuttle_positions['ball_hit'].iloc[i] = 1
                    
        frame_nums_with_ball_hits = df_shuttle_positions[df_shuttle_positions['ball_hit']==1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        shuttle_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                shuttle_detections = pickle.load(f)
            return shuttle_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            shuttle_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(shuttle_detections, f)
        
        return shuttle_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        shuttle_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            shuttle_dict[1] = result
            
        return shuttle_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, shuttle_dict in zip(video_frames, player_detections):
            # draw bounding box
            for track_id, bbox in shuttle_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Shuttle ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            output_video_frames.append(frame)

        return output_video_frames


