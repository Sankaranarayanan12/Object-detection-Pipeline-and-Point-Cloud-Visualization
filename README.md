# Object Detection Pipeline and Point Cloud Visualization
________________________
### Thought process and Workflow:
This project helped me get familiar with the visual stack for both 2-D and 3-D images/videos namely CV+YOLOv8n and Open3D.

#### 1) Object detection pipeline:
For this project I used the UA-DETRAC dataset which contains recorded videos of road traffic under different scenarios and I picked 3 different scenarios Sunny, Night and Rainy conditions.
First I started by picking ROI(Region of Interest) by letting the user choose a custom region from the image and taking those points for detection of objects done in the pick_roi.py file then I used the pretrained yolov8n model trained for object detection on the COCO classes to detect Car, bus and truck that fall inside
the ROI region by using the pointPolygonTest offered by OpenCV and accordingly coloring the detected boxes green and rest of the detections gray implemented in the detect.py file.
Finally in the evaluate.py file the predictions of the model were compared against the annotated XML files to get the average precision computed at confidence greater than equal to 0.001 and precision,recall,F1 scores computed at confidence greater than equal to 0.25 for both the ROI region and the full region. The evaluation was done based on IoU=0.5 and class-agnostic matching.
##### Result:
<img width="1171" height="530" alt="image" src="https://github.com/user-attachments/assets/8f438830-b179-4fa5-9674-0e693473451a" />
The average precision for the night condition in the ROI region drops as the ROI region chosen was a relatively small triangle and cases where multiple objects coincide with each other increases the error.

#### 2) Point Cloud Visualization:
For this project I used the Redwood RGB-D samples provided by Open3D library. I used the in-built function provided by Open3D which is o3d.geometry.PointCloud.create_from_depth_image() to convert the depth image to point clouds by passing the raw_depth image and the intrinsic parameters of the camera to the function.
I also performed a transformation of the output to get the correct orientation. Followed by this did the outlier removal, down sampling and coloring the regions based on proximity.
   

