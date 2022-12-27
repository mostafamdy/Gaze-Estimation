import math

import cv2
import mediapipe as mp
import numpy as np
import random
import pandas as pd

h = 600
w = 1200
blackWindow = np.zeros((h, w, 3), np.uint8)

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
counter=0


class Dataset:
    def __init__(self, head, left_iris, right_iris, left_eye, right_eye,centerX,centerY):
        self.head = head
        self.left_iris = left_iris
        self.right_iris = right_iris
        self.left_eye = left_eye
        self.right_eye = right_eye

        self.pointX = centerX
        self.pointY = centerY

    def to_dic(self):
        return {

            "yaw":self.head.yaw,
            "pitch":self.head.pitch,
            "roll":self.head.roll,

            "l_iris_center":self.left_iris.center,
            "l_iris_top":self.left_iris.top,
            "l_iris_bottom":self.left_iris.bottom,
            "l_iris_right":self.left_iris.right,
            "l_iris_left":self.left_iris.left,

            "r_iris_center": self.right_iris.center,
            "r_iris_top": self.right_iris.top,
            "r_iris_bottom": self.right_iris.bottom,
            "r_iris_right": self.right_iris.right,
            "r_iris_left": self.right_iris.left,

            "l_eye1":(self.left_eye[0].x,self.left_eye[0].y),
            "l_eye2":(self.left_eye[1].x,self.left_eye[1].y),
            "l_eye3":(self.left_eye[2].x,self.left_eye[2].y),
            "l_eye4":(self.left_eye[3].x,self.left_eye[3].y),
            "l_eye5":(self.left_eye[4].x,self.left_eye[4].y),
            "l_eye6":(self.left_eye[5].x,self.left_eye[5].y),
            "l_eye7":(self.left_eye[6].x,self.left_eye[6].y),
            "l_eye8":(self.left_eye[7].x,self.left_eye[7].y),
            "l_eye9":(self.left_eye[8].x,self.left_eye[8].y),
            "l_eye10":(self.left_eye[9].x,self.left_eye[9].y),
            "l_eye11":(self.left_eye[10].x,self.left_eye[10].y),
            "l_eye12":(self.left_eye[11].x,self.left_eye[11].y),
            "l_eye13":(self.left_eye[12].x,self.left_eye[12].y),
            "l_eye14":(self.left_eye[13].x,self.left_eye[13].y),
            "l_eye15":(self.left_eye[14].x,self.left_eye[14].y),
            "l_eye16":(self.left_eye[15].x,self.left_eye[15].y),

            "r_eye1": (self.right_eye[0].x,self.right_eye[0].y),
            "r_eye2": (self.right_eye[1].x,self.right_eye[1].y),
            "r_eye3": (self.right_eye[2].x,self.right_eye[2].y),
            "r_eye4": (self.right_eye[3].x,self.right_eye[3].y),
            "r_eye5": (self.right_eye[4].x,self.right_eye[4].y),
            "r_eye6": (self.right_eye[5].x,self.right_eye[5].y),
            "r_eye7": (self.right_eye[6].x,self.right_eye[6].y),
            "r_eye8": (self.right_eye[7].x,self.right_eye[7].y),
            "r_eye9": (self.right_eye[8].x,self.right_eye[8].y),
            "r_eye10": (self.right_eye[9].x,self.right_eye[9].y),
            "r_eye11": (self.right_eye[10].x,self.right_eye[10].y),
            "r_eye12": (self.right_eye[11].x,self.right_eye[11].y),
            "r_eye13": (self.right_eye[12].x,self.right_eye[12].y),
            "r_eye14": (self.right_eye[13].x,self.right_eye[13].y),
            "r_eye15": (self.right_eye[14].x,self.right_eye[14].y),
            "r_eye16": (self.right_eye[15].x,self.right_eye[15].y),

            "centerX":self.pointX,
            "centerY":self.pointY
        }

class HeadRotations:
    def __init__(self, pitch, yaw,roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class Iris:
    def __init__(self, center, top, bottom, right, left):
        self.center = center
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left


def getEularFromAxis(Axis):
    r00 = Axis[0][0]
    r01 = Axis[0][1]
    r02 = Axis[0][2]

    r10 = Axis[1][0]
    r11 = Axis[1][1]
    r12 = Axis[1][2]

    r20 = Axis[2][0]
    r21 = Axis[2][1]
    r22 = Axis[2][2]

    # thetaX=0
    # thetaY=0
    # thetaZ=0
    if r10 < 1:

        if r10 > -1:

            thetaZ = math.asin(r10)
            thetaY = math.atan2(-r20, r00)
            thetaX = math.atan2(-r12, r11)
        else:
            thetaZ = -math.PI / 2
            thetaY = -math.atan2(r21, r22)
            thetaX = 0


    else:
        thetaZ = math.PI / 2
        thetaY = math.atan2(r21, r22)
        thetaX = 0


    y = math.degrees(2 * -thetaX)
    z = math.degrees(2 * -thetaY)
    x = math.degrees(2 * -thetaZ)
    return np.array([y, z, x]);


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
isNext = False

blackWindow = np.zeros((h, w, 3), np.uint8)
randomX = random.randint(0, w - 10)
randomY = random.randint(0, h - 10)
# print(randomX,randomY)
cv2.circle(blackWindow, (randomX, randomY), 5, [255, 0, 0], -1)
result_image=np.zeros((h, w, 3), np.uint8)
data=[]
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # print(face_landmarks)
                # print("________________________________________")

                if isNext:
                    # dataset=Dataset()
                    result_image=image.copy()
                    # getting Euler Angles
                    top=face_landmarks.landmark[10]
                    bottom=face_landmarks.landmark[152]
                    left=face_landmarks.landmark[234]
                    right=face_landmarks.landmark[454]

                    y_axis=np.array([top.x-bottom.x,top.y-bottom.y,top.z-bottom.z])
                    x_axis=np.array([right.x-left.x,right.y-left.y,right.z-left.z])

                    normalized_y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
                    normalized_x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
                    z_axis = np.cross(normalized_x_axis,normalized_y_axis)
                    axis=[x_axis,y_axis,z_axis]

                    eulars=getEularFromAxis(axis)
                    headRot=HeadRotations(eulars[0],eulars[1],eulars[2])
                    left_iris=[]
                    print("image shape",image.shape)

                    for iris in LEFT_IRIS:
                       left_iris.append([int(face_landmarks.landmark[iris].x*image.shape[1]),int(face_landmarks.landmark[iris].y*image.shape[0])])


                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(np.array(left_iris))
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    cv2.circle(result_image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)

                    right_iris=[]
                    for iris in RIGHT_IRIS:
                       right_iris.append([int(face_landmarks.landmark[iris].x*image.shape[1]),int(face_landmarks.landmark[iris].y*image.shape[0])])

                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(np.array(right_iris))
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv2.circle(result_image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

                    left_ir = Iris((l_cx/w,l_cy/h),
                                   (face_landmarks.landmark[LEFT_IRIS[0]].x,face_landmarks.landmark[LEFT_IRIS[0]].y),
                                   (face_landmarks.landmark[LEFT_IRIS[1]].x,face_landmarks.landmark[LEFT_IRIS[1]].y),
                                   (face_landmarks.landmark[LEFT_IRIS[2]].x,face_landmarks.landmark[LEFT_IRIS[2]].y),
                                   (face_landmarks.landmark[LEFT_IRIS[3]].x,face_landmarks.landmark[LEFT_IRIS[3]].y),
                                   # face_landmarks.landmark[LEFT_IRIS[1]],
                                   # face_landmarks.landmark[LEFT_IRIS[2]],
                                   # face_landmarks.landmark[LEFT_IRIS[3]]
                                   )

                    right_ir = Iris((r_cx/w,r_cy/h),
                                    (face_landmarks.landmark[RIGHT_IRIS[0]].x, face_landmarks.landmark[RIGHT_IRIS[0]].y),
                                    (face_landmarks.landmark[RIGHT_IRIS[1]].x, face_landmarks.landmark[RIGHT_IRIS[1]].y),
                                    (face_landmarks.landmark[RIGHT_IRIS[2]].x, face_landmarks.landmark[RIGHT_IRIS[2]].y),
                                    (face_landmarks.landmark[RIGHT_IRIS[3]].x, face_landmarks.landmark[RIGHT_IRIS[3]].y),

                                    # face_landmarks.landmark[RIGHT_IRIS[0]],
                                    #            face_landmarks.landmark[RIGHT_IRIS[1]], face_landmarks.landmark[RIGHT_IRIS[2]],
                                    #            face_landmarks.landmark[RIGHT_IRIS[3]]
                                    )


                    right_eye = []
                    left_eye=[]

                    for eye in RIGHT_EYE:
                        right_eye.append(face_landmarks.landmark[eye])
                        cv2.circle(result_image,(int(face_landmarks.landmark[eye].x*image.shape[1]),int(face_landmarks.landmark[eye].y*image.shape[0])),1,[255,0,255],1)

                    for eye in LEFT_EYE:
                        left_eye.append(face_landmarks.landmark[eye])
                        cv2.circle(result_image, (int(face_landmarks.landmark[eye].x * image.shape[1]),
                                           int(face_landmarks.landmark[eye].y * image.shape[0])), 1, [255, 0, 255], 1)

                    data.append(Dataset(headRot,left_ir,right_ir,left_eye,right_eye,randomX,randomY))
                    # data.append(Dataset(headRot.__dict__,left_ir.__dict__,right_ir.__dict__,left_eye,right_eye,randomX,randomY))
                    # print(data[0].__dict__)
                    cv2.putText(result_image,"pitch "+str(int(eulars[0])),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    cv2.putText(result_image,"yaw "+str(int(eulars[1])),(50, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    cv2.putText(result_image,"roll "+str(int(eulars[2])),(50, 150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                    cv2.circle(result_image, (int(top.x*image.shape[1]),int(top.y*image.shape[0])), 5, [255, 0, 0], -1)
                    cv2.circle(result_image, (int(bottom.x*image.shape[1]),int(bottom.y*image.shape[0])), 5, [255, 0, 0], -1)
                    cv2.circle(result_image, (int(left.x*image.shape[1]),int(left.y*image.shape[0])), 5, [255, 0, 0], -1)
                    cv2.circle(result_image, (int(right.x*image.shape[1]),int(right.y*image.shape[0])), 5, [255, 0, 0], -1)
                    cv2.putText(result_image,"count "+str(counter),(50, 200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                    blackWindow = np.zeros((h, w, 3), np.uint8)
                    randomX = random.randint(0, w - 10)
                    randomY = random.randint(0, h - 10)
                    print("random XY")
                    print(randomX, randomY)
                    cv2.circle(blackWindow, (randomX, randomY), 5, [255, 0, 0], -1)
                    isNext = False
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)
        cv2.imshow('BlackWindow', blackWindow)
        cv2.imshow('result', result_image)

        k = cv2.waitKey(5)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            df = pd.DataFrame([t.to_dic() for t in data])
            print("_____________________________________________")
            print("df")
            print(df)
            # df['left_iris'] = df['left_iris'].map(lambda x: ','.join(map(str, x)))
            df.to_csv("data.csv",mode='a', header=False)
            break
        elif k == ord('n'):
            print("next")
            counter+=1
            isNext = True
        # isNext = True
cap.release()

