# Object Detection Pipeline and Point Cloud Visualization
________________________
### Thought process and Workflow:
This project helped me get familiar with the visual stack for both 2-D and 3-D images/videos namely CV+YOLOv8 and Open3D.

#### 1) Object detection pipeline:
For this project I used the UA-DETRAC dataset which contains recorded videos of road traffic under different scenarios and I picked 3 different scenarios Sunny, Night and Rainy conditions.
First I started by picking ROI(Region of Interest) by letting the user choose a custom region from the image and taking those points for detection of objects(using OpenCV) done in the pick_roi.py file then I used the pretrained YOLOV8 model trained for object detection on the COCO classes to detect Car, motorcycle, bus and truck that fall inside
the ROI region by using the pointPolygonTest offered by OpenCV and accordingly coloring the detected boxes green and rest of the detections gray implemented in the detect.py file.
Finally in the evaluvate.py file the predictions of the model was compared against the annotated XML files to get the average precision,precision,recall,F1 scores, inference time for both the ROI region and the full region. 
##### Result:
<img width="1171" height="530" alt="image" src="https://github.com/user-attachments/assets/8f438830-b179-4fa5-9674-0e693473451a" />
From the F1 scores we can infer that the model tends to perform well in the full region in the rainy condition than in the sunny and the night conditions.

#### 2) Point Cloud Visualization:
For this project I used the Redwood RGB-D samples provided by Open3D library. I used the in-built function provided by Open3D which is o3d.geometry.PointCloud.create_from_depth_image() to convert the depth images to point clouds by passing the raw_depth image directory and the intrinsic parameters of the camera.
I also performed a transformation of the output to get the correct orientation. Followed by this did the outlier removal, down sampling and coloring the regions based on proximity.
   

