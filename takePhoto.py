import cv2, time


def addUser():
    name = raw_input("Enter name of user to be registered:")
    print("Preparing to take user ID photo.\n"
          "Please face forward, with eyes directed at center of screen.\n"
          "Press \"SPACE\" to capture photo or \"ESC\" to quit.")

    toPhoto = raw_input("Press \"ENTER\" now to proceed.")

    if toPhoto == "":
        getPhoto(name)




def getPhoto(stringName):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("addUser")
    while True:
        ret, frame = cam.read()
        cv2.imshow("addUser", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "user {}.png".format(stringName)
            img_path = "photo/" + img_name
            cv2.imwrite(img_path, frame)
            print("{} has been added to list of verified users.".format(stringName))
            break
    cam.release()

    cv2.destroyAllWindows()


addUser()