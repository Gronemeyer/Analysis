import cv2
cap = cv2.VideoCapture(r'C:\Users\SIPE_LAB\Downloads\240701_sb7_grat_6_DLCDLC_Resnet50_pupil_testJul29shuffle2_snapshot_200_p60_labeled.mp4')
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800,800)

while(cap.isOpened()):
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, 765)
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()