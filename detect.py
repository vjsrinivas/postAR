import cv2
import numpy as np
from posters import PosterType

def colorMasking(frame, low_color, high_color):
    h,w,c = frame.shape
    mask = cv2.inRange(frame, low_color, high_color).astype(np.float32)/255
    
    # convert mask to quadrilateral
    kernel = 20
    _convolved = cv2.filter2D(mask, -1, np.ones((kernel, kernel), np.float32))
    (T, _thresh) = cv2.threshold(_convolved, 1, 255, cv2.THRESH_BINARY)

    #_re_convolved = cv2.resize(_convolved, (_convolved.shape[1]//kernel, _convolved.shape[0]//kernel))
    # find x1y1 (top,left), x2y2 (bottom,right)
    indices = np.where(_thresh[:,:] > 0)

    # debug:
    _new_image = np.ndarray((h,w,c))
    _new_image[:,:,0] = _convolved; _new_image[:,:,1] = _convolved; _new_image[:,:,2] = _convolved
    x1,y1,x2,y2,x3,y3,x4,y4 = -1,-1,-1,-1,-1,-1,-1,-1

    if len(indices) > 0:
        if len(indices[0]) > 4:
            least_x, most_x = np.min(indices[1]), np.max(indices[1])
            least_y, most_y = np.min(indices[0]), np.max(indices[0])
            
            '''
            nx1 = np.min(indices[1])
            ny1 = indices[0][np.argmin(indices[1])]
            nx1 = (least_x + nx1)//2
            ny1 = (least_y+ny1)//2

            nx3 = np.max(indices[1])
            ny3 = indices[0][np.argmax(indices[1])]
            nx3 = (most_x + nx3)//2
            ny3 = (most_y+ny3)//2
            '''

            # cut up into four quadrants:
            x1,y1 = least_x, least_y
            x2,y2 = most_x, least_y
            x3,y3 = most_x, most_y,
            x4,y4 = least_x, most_y
            
            '''
            q1 = ((x1,y1), (x1+((x2-x1)//2), y1+((y4-y1)//2) ))
            q2 = ((x1+(x2-x1)//2, y1), (x2, (y1+(y3-y1)//2)))
            q3 = ( (x1+((x2-x1)//2), y1+(y3-y1)//2), (x3,y3))
            q4 = ((x1,y1+((y4-y1)//2)), (x1+((x2-x1)//2),y4))

            q2_indices_x = np.where( np.logical_and(indices[1] >= q2[0][0], indices[1] <= q2[1][0]) )[0]
            q2_indices_y = np.where( np.logical_and(indices[0] >= q2[0][1], indices[0] <= q2[1][1]) )[0]
            print(q2_indices_x.shape, q2_indices_y.shape)
            q2_indices_y = indices[0][q2_indices_y]
            q2_indices_x = indices[1][q2_indices_x]
            
            q4_indices_x = np.where( np.logical_and(indices[1] >= q4[0][0], indices[1] <= q4[1][0]) )[0]
            q4_indices_y = np.where( np.logical_and(indices[0] >= q4[0][1], indices[0] <= q4[1][1]) )[0]
            q4_indices_y = indices[0][q4_indices_y]
            q4_indices_x = indices[1][q4_indices_x]

            nx1, ny1 = np.max(q1_indices_x), np.min(q1_indices_y)
            print(nx1,ny1)
            _new_image = cv2.circle(_new_image, (nx1,ny1), 10, (255,0,0), -1)

            nx2, ny2 = np.max(q2_indices_x), np.min(q2_indices_y)
            nx4, ny4 = np.min(q4_indices_x), np.max(q4_indices_y) 
            '''

            # debugging:
            
            for xy in [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]:
                x,y = xy
                _new_image = cv2.circle(_new_image, (x,y), 3, (0,255,0), -1)
            _new_image = cv2.rectangle(_new_image, (x1, y1), (x3,y3), (0,0,255), 2)
            

            '''
            _new_image = cv2.circle(_new_image, (nx1,ny1), 5, (255,0,0), -1)
            _new_image = cv2.circle(_new_image, (nx3,ny3), 5, (255,0,0), -1)
            _new_image = cv2.circle(_new_image, (nx2,ny2), 5, (255,0,0), -1)
            _new_image = cv2.circle(_new_image, (nx4,ny4), 5, (255,0,0), -1)
            '''

    rect = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    return _new_image, rect

def projectionVector(frame, rect, poster):
    # right now, just project as straight rectangle:
    # later figure out prespective
    # rect: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (clockwise points starting from top-left)
    #print(type(poster)==cv2.VideoCapture, type(poster))
    if type(poster) == np.ndarray:
        flag = PosterType.IMAGE
    elif type(poster) == cv2.VideoCapture:
        flag = PosterType.VIDEO
        ret, vid_frame = poster.read()
        if not ret: 
            poster.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, vid_frame = poster.read()
            if not ret:
                print('bad video file!')
                return frame
    else:
        flag = PosterType.THREED

    if not any(-1 in sl for sl in rect):
        x1,y1 = rect[0]
        x3,y3 = rect[2]
        center_pt = [(x1+x3)//2, (y1+y3)//2]
        h,w = y3-y1, x3-x1 # temp
        if flag == PosterType.IMAGE:
            re_poster = cv2.resize(poster, (w,h))
        elif flag == PosterType.VIDEO:
            re_poster = cv2.resize(vid_frame, (w,h))
        frame[y1:y3, x1:x3, :] = re_poster
    return frame

def cornerMasking(frame):
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    greyFrame = np.float32(greyFrame)
    dst = cv2.cornerHarris(greyFrame,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(greyFrame,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    for c in range(res.shape[0]):
        frame = cv2.circle(frame, (res[c,2 ], res[c,3]), 3, (0,0,255), -1)
        frame = cv2.circle(frame, (res[c,0], res[c,1]), 3, (0,255,0), -1)

    #rame[res[:,1],res[:,0]]=[0,0,255]
    #frame[res[:,3],res[:,2]] = [0,255,0]
    return frame

def barrelDistort(frame):
    # make two images
    # shift left one 

    height, width, channel = frame.shape
    distCoeff = np.zeros((4,1),np.float64)

    # TODO: add your coefficients here!
    k1 = 1.0e-5; # negative to remove barrel distortion
    k2 = 0.5e-7
    p1 = 0.5e-7
    p2 = 0.5e-7
    
    #k2 = 1.0e-7
    #p1 = 1.0e-7
    #p2 = 1.0e-7

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y

    # here the undistortion will be computed
    dst = cv2.undistort(frame,cam,distCoeff)
    return dst

def lens(frame):
    h,w,c = frame.shape
    new_frame = np.hstack((frame, frame))
    #new_frame = np.ndarray((h, w*2, 3)).astype(np.uint8)
    #new_frame[0:h,0:w,:] = frame
    #new_frame[0:h, w:, :] = frame
    return new_frame

if __name__ == '__main__':
    '''
    img = cv2.imread('test_image.png')
    cv2.imshow('corner', cornerMasking(img) )
    cv2.waitKey(-1)
    exit()
    '''

    #[ 33  86 157] [ 53 106 237]
    #[ 33  86 159] [ 53 106 239]
    #[ 33  83 163] [ 53 103 243]
    low_range = np.array([33, 83, 163])
    high_range = np.array([53, 103, 243])
    
    img = cv2.imread('test_image.png')
    h,w,c = img.shape
    center_img = img[(h//2)-50:(h//2)+50, (w//2)-50:(w//2)+50,:]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = colorMasking(hsv_img, low_range, high_range)

    cv2.imshow('mask', mask)
    cv2.imshow('rgb', img)
    #cv2.imshow('center', center_img)
    #print(cv2.cvtColor(center_img, cv2.COLOR_BGR2HSV))
    cv2.waitKey(-1)