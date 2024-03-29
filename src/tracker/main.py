from tracker import *
import argparse


confirm_box = False
draw_rectangle = False
box_x = 0
box_y = 0
box_w = 0
box_h = 0


#OpenCV callback function to get the area of the user selected object
def selectTarget(event, x, y, flags, param):
    global box_x, box_y, box_w, box_h, confirm_box, draw_rectangle

    if event == cv2.EVENT_MOUSEMOVE:
        if(draw_rectangle):
            box_w = x - box_x
            box_h = y - box_y

    elif event == cv2.EVENT_LBUTTONDOWN:
        draw_rectangle = True
        box_x = x
        box_y = y
        box_w = 0
        box_h = 0

    elif event == cv2.EVENT_LBUTTONUP:
        draw_rectangle = False
        if box_w < 0:
            box_x += box_w
        if box_h < 0:
            box_y += box_h
            box_h *= -1
        pt1 = (box_x, box_y)
        pt2 = (box_x + box_w, box_y + box_h)
        cv2.rectangle(param, pt1, pt2, (0, 255, 0), 2)
        confirm_box = True
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="path to the video to be analyzed", required=True)
    parser.add_argument("-np", "--nb_particles", type=int, help="number of particles to be used for the tracking", required=False)
    parser.add_argument("-op", "--object_speed", type=int, help="object speed (used to predict object motion)", required=False)
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)

    ret, frame = video.read()
    cv2.namedWindow("Tracker")

    tmp = frame.copy()
    cv2.setMouseCallback("Tracker", selectTarget, tmp)
    while not confirm_box:
        tmp2 = tmp.copy()
        if draw_rectangle:
            pt1 = (box_x, box_y)
            pt2 = (box_x + box_w, box_y + box_h)
            cv2.rectangle(tmp2, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("Tracker", tmp2)
        if cv2.waitKey(30) == ord('c'):
            break

    tracker = Tracker(frame, (box_x, box_y, box_w, box_h), args.nb_particles, args.object_speed)
    tracker.ParticlesInitilization()
    tracker.GetTargetHistogram()
    while 1:
        ret, frame = video.read()
        if not ret:
            break
            
        tracker.ParticlesResampling()
        tracker.ParticlesMotionModel()
        tracker.ParticlesAppearanceModel(frame)
        tracker.UpdateParticlesWeight()
        tracker.UpdateTargetPosition()
        tracker.DrawParticles(frame)
        tracker.DrawTargetBox(frame)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(30) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()