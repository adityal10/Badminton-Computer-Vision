from utils import read_video, save_video, measure_distance, convert_pixel_distance_to_meters, draw_player_stats
from trackers import PlayerTracker, ShuttleTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants

import cv2
import pickle
from copy import deepcopy
import pandas as pd
import streamlit as st
import os
import subprocess

import warnings
warnings.filterwarnings("ignore")

import moviepy.editor as moviepy

save_folder = 'input_videos'
os.makedirs(save_folder, exist_ok=True)


def main():
    st.title("Player Detection - Badminton")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        save_path = os.path.join(save_folder, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # input video path (read video)
        # input_video_path = 'input_videos/input_video_10s.mp4'
        st.write("Detecting video frames")
        video_frames = read_video(save_path)
        st.success('Detection Success!', icon="✅")

        st.write("Detecting player and suttle")
        # detecting player detections and shuttle detectors
        player_tracker = PlayerTracker(model_path='yolov8x.pt')
        shuttle_tracker = ShuttleTracker(model_path='models/shuttle_detect_model.pt')


        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')
        shuttle_detections = shuttle_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/shuttle_detections.pkl')

        shuttle_detections = shuttle_tracker.interpolate_shuttle_positions(shuttle_detections)
        # print(player_detections)
        st.success('Detection Success!', icon="✅")


        st.write("Court Keypoints detection")
        # court line detection model
        court_model_path = 'models/court_kps_model.pt'
        court_line_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_line_detector.predict(video_frames[0])
        st.success('Detection Success!', icon="✅")

        st.write("Choosing players")
        # chose players
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        st.success('Detection Success!', icon="✅")

        # st.write("Minicourt")
        # # minicourt
        # mini_court = MiniCourt(video_frames[0])
        # st.success('Detection Success!', icon="✅")

        st.write("Detect shuttle shorts")
        # detect ball shorts
        ball_shot_frames = shuttle_tracker.get_ball_shot_frames(shuttle_detections)
        # print(ball_shot_frames)
        st.success('Detection Success!', icon="✅")

        # st.write("Convering position to mini court positions")
        # # covnert the positions to mini court positions
        # player_mini_court_detections, shuttle_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, shuttle_detections, court_keypoints)
        # st.success('Converting Success!', icon="✅")


        # st.write("Player Stats")
        # player_stats_data = [{
        #     'frame_num':0,
        #     'player_1_number_of_shots':0,
        #     'player_1_total_shot_speed':0,
        #     'player_1_last_shot_speed':0,
        #     'player_1_total_player_speed':0,
        #     'player_1_last_player_speed':0,

        #     'player_2_number_of_shots':0,
        #     'player_2_total_shot_speed':0,
        #     'player_2_last_shot_speed':0,
        #     'player_2_total_player_speed':0,
        #     'player_2_last_player_speed':0,

        #     'player_11_number_of_shots':0,
        #     'player_11_total_shot_speed':0,
        #     'player_11_last_shot_speed':0,
        #     'player_11_total_player_speed':0,
        #     'player_11_last_player_speed':0,

        #     'player_12_number_of_shots':0,
        #     'player_12_total_shot_speed':0,
        #     'player_12_last_shot_speed':0,
        #     'player_12_total_player_speed':0,
        #     'player_12_last_player_speed':0,
        # } ]

        # for ball_shot_ind in range(len(ball_shot_frames)-1):
        #     start_frame = ball_shot_frames[ball_shot_ind]
        #     end_frame = ball_shot_frames[ball_shot_ind+1]
        #     ball_shot_time_in_secs = (end_frame-start_frame)/24 # 24fps
        #     # get distanc covered by the ball
        #     distance_covered_by_ball_in_pixels = measure_distance(shuttle_mini_court_detections[start_frame][1],
        #                                                             shuttle_mini_court_detections[end_frame][1])
        #     distance_covered_by_ball_in_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_in_pixels,
        #                                                                             constants.DOUBLE_LINE_WIDTH,
        #                                                                             mini_court.get_width_of_mini_court())
        #     # speed of the ball in km/h
        #     speed_of_ball_shot = distance_covered_by_ball_in_meters/ball_shot_time_in_secs*3.6
        #     # player who shot the ball
        #     player_positions = player_mini_court_detections[start_frame]
        #     player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
        #     shuttle_mini_court_detections[start_frame][1]))
        #     # opponent player spped
        #     opponent_player_id = 1 if player_shot_ball == 2 else 2
        #     distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
        #     player_mini_court_detections[end_frame][opponent_player_id])
        #     distance_covered_by_opponent_in_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_in_pixels,
        #                                                                             constants.DOUBLE_LINE_WIDTH,
        #                                                                             mini_court.get_width_of_mini_court())
        #     # speed of the opponent
        #     speed_of_opponent = distance_covered_by_opponent_in_meters/ball_shot_time_in_secs*3.6
        #     current_player_stats = deepcopy(player_stats_data[-1])
        #     current_player_stats['frame_num'] = start_frame
        #     current_player_stats[f'player_{player_shot_ball}_number_of_shots'] +=1
        #     current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        #     current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] += speed_of_ball_shot
            
        #     current_player_stats[f'player_{opponent_player_id}_last_shot_speed'] += speed_of_opponent
        #     current_player_stats[f'player_{opponent_player_id}_last_shot_speed'] += speed_of_opponent
        #     player_stats_data.append(current_player_stats)
                
        # player_stats_data_df = pd.DataFrame(player_stats_data)
        # frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        # player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        # player_stats_data_df = player_stats_data_df.ffill()

        # player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
        # player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
        # player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
        # player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']
        # st.success('Stats Success!', icon="✅")


        st.write("Output Video in making")
        ## draw output
        # draw player bounding box
        output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
        output_video_frames = shuttle_tracker.draw_bboxes(output_video_frames, shuttle_detections)

        # draw court keypoints
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

        # # draw mini court
        # output_video_frames = mini_court.draw_mini_court(output_video_frames)
        # output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, positions=player_mini_court_detections,color=(0,255,255))
        # output_video_frames = mini_court.draw_shuttle_points_on_mini_court(output_video_frames, positions=shuttle_mini_court_detections,color=(0,255,255))


        # draw player stats
        # output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

        # draw frame num on top left corner
        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


        output_file_name = 'output_videos/output_video.avi'
        save_video(output_video_frames, output_file_name)
        # st.download_button(label='outputvideo', data=output_file_name)
        st.success('Outcome Video Success!', icon="✅")

        # converts .avi file mp4
        clip = moviepy.VideoFileClip(output_file_name)

        # saves the mp4 file to the output videos folder
        output_file_name_mp4 = output_file_name.replace('avi','mp4')
        clip.write_videofile(output_file_name_mp4)

        # displays the output video in the main page
        st.video(output_file_name_mp4, autoplay=True)

        # downloads the video
        # Read the video file
        with open(output_file_name_mp4, "rb") as file:
            video_bytes = file.read()

        # Create a download button for the video
        st.download_button(label="Download Video",data=video_bytes,file_name="video.mp4",mime="video/mp4")


if __name__=="__main__":
    main()