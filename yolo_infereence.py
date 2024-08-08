from ultralytics import YOLO

model = YOLO('models/court_kps_model.pt')
result = model.predict('input_videos/image1.png',conf=0.2)
# print(result[0])
print(result[0].keypoints.xy.squeeze().cpu().numpy().flatten())

# for r in result:
#     print(r.keypoints)

# print(result)
# print("boxes")
# for box in result[0].boxes:
#     print(box)