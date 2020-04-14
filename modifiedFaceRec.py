import face_recognition
import cv2
import numpy as np
import os


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# for every photo in photo folder, map the face to an encoding and create a registered user
def verifyUser(reporting=True, headTiltCheck=True, eyeDirectionCheck=True, lightThresholdVar=65):
    known_face_encodings = []
    known_face_names = []
    path = os.getcwd()
    print("starting")
    imgNames = os.listdir(path + "/photo")
    for x in imgNames:
        usrImg = face_recognition.load_image_file("photo/" + x)
        usrFaceEncoding = face_recognition.face_encodings(usrImg)[0]
        usrName = x.replace("user ", "").replace(".jpg", "").replace(".png", "")
        known_face_encodings.append(usrFaceEncoding)
        known_face_names.append(usrName)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    if eyeDirectionCheck:
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        detector = cv2.SimpleBlobDetector_create(detector_params)

    verifiedUser = False
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if not verifiedUser:

            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, model="small")
                    if headTiltCheck:
                        leftEyeCenter = ((((face_landmarks_list[0]["left_eye"][0][0]) +
                                           (face_landmarks_list[0]["left_eye"][1][0])) * 2),
                                         (((face_landmarks_list[0]["left_eye"][0][1]) +
                                           (face_landmarks_list[0]["left_eye"][1][1])) * 2))
                        rightEyeCenter = ((((face_landmarks_list[0]["right_eye"][0][0]) +
                                            (face_landmarks_list[0]["right_eye"][1][0])) * 2),
                                          (((face_landmarks_list[0]["right_eye"][0][1]) +
                                            (face_landmarks_list[0]["right_eye"][1][1])) * 2))
                        nosePointCenter = ((face_landmarks_list[0]["nose_tip"][0][0]) * 4,
                                           (face_landmarks_list[0]["nose_tip"][0][1]) * 4)

                        distLeftEyetoNose = np.linalg.norm(np.array(leftEyeCenter) - np.array(nosePointCenter))
                        distRightEyetoNose = np.linalg.norm(np.array(rightEyeCenter) - np.array(nosePointCenter))
                        threshold = 0.05
                        ratio = abs(1 - (distLeftEyetoNose / distRightEyetoNose))
                        faceForward = (ratio <= threshold)
                    else:
                        faceForward = True

                    if eyeDirectionCheck:

                        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        xboxLeft1 = face_landmarks_list[0]["left_eye"][0][0] * 4
                        xboxLeft2 = face_landmarks_list[0]["left_eye"][1][0] * 4
                        yboxLeft1 = max(face_landmarks_list[0]["left_eye"][0][1] * 4,
                                        face_landmarks_list[0]["left_eye"][1][1] * 4) + abs(xboxLeft1 - xboxLeft2)/3
                        yboxLeft2 = min(face_landmarks_list[0]["left_eye"][0][1] * 4,
                                        face_landmarks_list[0]["left_eye"][1][1] * 4) - abs(xboxLeft1 - xboxLeft2)/3
                        leftEyeImg = greyFrame[yboxLeft2:yboxLeft1, xboxLeft1:xboxLeft2]

                        xboxRight1 = face_landmarks_list[0]["right_eye"][1][0] * 4
                        xboxRight2 = face_landmarks_list[0]["right_eye"][0][0] * 4
                        yboxRight1 = max(face_landmarks_list[0]["right_eye"][0][1] * 4,
                                        face_landmarks_list[0]["right_eye"][1][1] * 4) + abs(xboxLeft1 - xboxLeft2)/3
                        yboxRight2 = min(face_landmarks_list[0]["right_eye"][0][1] * 4,
                                        face_landmarks_list[0]["right_eye"][1][1] * 4) - abs(xboxLeft1 - xboxLeft2)/3
                        rightEyeImg = greyFrame[yboxRight2:yboxRight1, xboxRight1:xboxRight2]

                        centerVar = 4

                        leftEyeImg = cv2.threshold(leftEyeImg, lightThresholdVar, 255, cv2.THRESH_BINARY)[1]
                        leftEyeImg = cv2.erode(leftEyeImg, None, iterations=2)
                        leftEyeImg = cv2.dilate(leftEyeImg, None, iterations=4)
                        leftEyeImg = cv2.medianBlur(leftEyeImg, 5)
                        keypointsLeft = detector.detect(leftEyeImg)
                        if len(keypointsLeft) != 0:
                            trueLeftPupil = (int(round(keypointsLeft[0].pt[0], 2) + xboxLeft1), int(round(keypointsLeft[0].pt[1], 4) + yboxLeft2))
                            isLCentered = abs((abs(xboxLeft1 - xboxLeft2)/2) - round(keypointsLeft[0].pt[0], 0)) <= centerVar
                        else:
                            trueLeftPupil = (-4, -4)
                            isLCentered = False

                        rightEyeImg = cv2.threshold(rightEyeImg, lightThresholdVar, 255, cv2.THRESH_BINARY)[1]
                        rightEyeImg = cv2.erode(rightEyeImg, None, iterations=2)
                        rightEyeImg = cv2.dilate(rightEyeImg, None, iterations=4)
                        rightEyeImg = cv2.medianBlur(rightEyeImg, 5)
                        keypointsRight = detector.detect(rightEyeImg)
                        if len(keypointsRight) != 0:
                            trueRightPupil = (int(round(keypointsRight[0].pt[0], 2) + xboxRight1), int(round(keypointsRight[0].pt[1], 4) + yboxRight2))
                            isRCentered = abs((abs(xboxRight1 - xboxRight2) / 2) - round(keypointsRight[0].pt[0])) <= centerVar
                        else:
                            trueRightPupil = (-4, -4)
                            isRCentered = False





                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    if 100 * (round((1 - face_distances[best_match_index]), 4)) >= 60:
                        userRecognized = True
                    else:
                        userRecognized = False

                    face_names.append(name)

            process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            font = cv2.FONT_HERSHEY_DUPLEX
            if headTiltCheck:
                # Display eye locations
                cv2.circle(frame, leftEyeCenter, 8, (0, 255, 0))
                cv2.circle(frame, rightEyeCenter, 8, (0, 255, 0))
                if not faceForward:
                    cv2.putText(frame, "Face forward.", (200, 20), font, 0.7, (0, 0, 255), 1)
                if reporting:
                    # Display distances for head tilt check
                    cv2.putText(frame, str(round(distRightEyetoNose, 2)), (80, 20), font, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, str(round(distLeftEyetoNose, 2)), (10, 20), font, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, str(round((ratio), 2)), (400, 20), font, 0.6, (255, 255, 255), 1)

            if eyeDirectionCheck:
                cv2.rectangle(frame, (xboxLeft1, yboxLeft1), (xboxLeft2, yboxLeft2), (255, 255, 255), 1)
                cv2.circle(frame, trueLeftPupil, 5, (255, 180, 100), 2)
                cv2.circle(frame, trueRightPupil, 5, (255, 180, 100), 2)
                if reporting:
                    cv2.putText(frame, str(isLCentered), (10, 50), font, 0.5, (255, 180, 100), 1)
                    cv2.putText(frame, str(isRCentered), (70, 50), font, 0.5, (255, 180, 100), 1)
            if reporting:
                similarity = str(100 * (round((1 - face_distances[best_match_index]), 4)))
                cv2.putText(frame, similarity, (500, 20), font, 0.6, (255, 255, 255), 1)

            # Display box around face, and label user if verified
            verifiedUser = (userRecognized and faceForward and isLCentered and isRCentered)

            if verifiedUser:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
                toDisplay = "VERIFIED: " + name
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, toDisplay, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

verifyUser()