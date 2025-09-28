import cv2
import numpy as np
from object_detection import ObjectDetection
import math
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N ", count," " , x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    if count <=2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                 tracking_objects[track_id] = pt
                 track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
                object_exists= False
                for pt in center_points_cur_frame_copy:
                 distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                
                #update ids position 
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame_copy.remove(pt)
                    continue

                #remove lost ids
                if not object_exists:
                    tracking_objects.pop(object_id)
       
       #add new ids 
        for pt in center_points_cur_frame:
                    tracking_objects[object_id]= pt 
                    track_id +=1
                

        for object_id, pt2 in tracking_objects.items():
                  cv2.circle(frame, pt2, 5, (0, 0, 255), -1)
                  cv2.putText(frame, str(object_id), (pt2[0], pt[1] - 7),0,1,(0, 0, 255),2)
        

   
    print("TRACKING OBJECTS")
    print(tracking_objects)

    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)


    #for pt in center_points_cur_frame:
        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()


    key = cv2.waitKey(1)
    if key == 27:
        break
