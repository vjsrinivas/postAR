import cv2
import depthai as dai
from detect import colorMasking, cornerMasking, barrelDistort, lens, projectionVector
import numpy as np
from posters import POSTERS
import time
import imutils

cat_poster = POSTERS.init(0)
cat_video = POSTERS.init(1)
#cv2.imshow('test', cat_poster)
#cv2.waitKey(-1)

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
margins = (50,50)
videoSize = (640,480)
camRgb.setVideoSize(videoSize[0]+(margins[0]*2), videoSize[1]+(margins[1]*2))

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)

# Hardset color ranges:
#[ 36 122  74] [ 56 142 154]
low_range = np.array([36, 122, 74])
high_range = np.array([56, 142, 154])

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        t1 = time.time()

        videoIn = video.get()
        rgbFrame = videoIn.getCvFrame()

        #yuvFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        #yuvFrame[:,:,0] = cv2.equalizeHist(yuvFrame[:,:,0])
        # convert the YUV image back to RGB format
        #rgbFrames = cv2.cvtColor(yuvFrame, cv2.COLOR_YUV2BGR)
        hsvFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2HSV)

        #out = cornerMasking(rgbFrame)
        out, rect  = colorMasking(hsvFrame, low_range, high_range)
        rgbFrame = projectionVector(rgbFrame, rect, cat_video)
        rgbFrame = barrelDistort(rgbFrame)
        finalFrame = lens(rgbFrame)
        finalFrame = imutils.resize(finalFrame, width=1400)

        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        #cv2.imshow("video", finalFrame)
        cv2.imshow("video", rgbFrame)
        cv2.imshow("mask", out)
        t2 = time.time()
        print("Inference time: %f"%(1/(t2-t1)))

        if cv2.waitKey(1) == ord('q'):
            break
