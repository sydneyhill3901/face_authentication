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
known_face_encodings = []
known_face_names = []
path = os.getcwd()
print(path)
imgNames = os.listdir(path + "/photo")
print(imgNames)
for x in imgNames:
    usrImg = face_recognition.load_image_file("photo/" + x)
    usrFaceEncoding= face_recognition.face_encodings(usrImg)[0]
    usrName = x.replace("user ", "").replace(".jpg", "").replace(".png","")
    known_face_encodings.append(usrFaceEncoding)
    known_face_names.append(usrName)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
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
                leftEyeCenter = ((((face_landmarks_list[0]["left_eye"][0][0]) +
                                (face_landmarks_list[0]["left_eye"][1][0]))*2),
                                (((face_landmarks_list[0]["left_eye"][0][1]) +
                                (face_landmarks_list[0]["left_eye"][1][1]))*2))
                rightEyeCenter = ((((face_landmarks_list[0]["right_eye"][0][0]) +
                                (face_landmarks_list[0]["right_eye"][1][0]))*2),
                                (((face_landmarks_list[0]["right_eye"][0][1]) +
                                (face_landmarks_list[0]["right_eye"][1][1]))*2))
                nosePointCenter = ((face_landmarks_list[0]["nose_tip"][0][0])*4,
                                (face_landmarks_list[0]["nose_tip"][0][1])*4)

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if 100 * (round((1 - face_distances[best_match_index]), 4)) >= 75:
                    verifiedUser = True

                face_names.append(name)

        process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        print(rightEyeCenter)

        # Display eye locations
        cv2.circle(frame, leftEyeCenter, 3, (0, 255, 0))
        cv2.circle(frame, rightEyeCenter, 3, (0, 255, 0))

        # Display box around face, and label user if verified
        font = cv2.FONT_HERSHEY_DUPLEX
        percentSimilarity = str(100 *(round((1 - face_distances[best_match_index]), 4))) + "%"
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