import cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
import warnings
import face_alignment
warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':

    # Load the video file and count number of frames
    cam = cv2.VideoCapture(0)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    print("Tracking started...")

    while (True):

        ret, curr_frame = cam.read()

        preds = fa.get_landmarks(curr_frame)

        print(preds)

        '''

        # Measurement, prediction and update

        # ***Task for you*** (0.5 points)
        # Make a Kalman prediction here: 
        predicted_state = kalman_predict(kal_A, kal_x)
        print(predicted_state)

        # Object detection (measurement)
        try:
            measured_position = object_detect(frame_current)

            center_coordinates = (int(np.round(measured_position[1])),
                                  int(np.round(measured_position[0])))

            cv2.circle(frame_current, center_coordinates, 10, (0, 255, 0), 2)
            center_coordinates_label = tuple(map(sum, zip(center_coordinates, (20,0))))
            cv2.putText(frame_current,    # image
                'detected',  # text
                center_coordinates_label,    # start position
                font,       # font
                0.5,        # size
                (0, 255, 0),# BGR color
                1,          # thickness
                cv2.LINE_AA) # type of line

        except ValueError:
            
            # ***Task for you*** (1 point)
            # If we do not have a measurement for this frame, let's use the _predicted_ state:
            measured_position = np.zeros(2) # change this!
            measured_position[0] = predicted_state[0, 0]
            measured_position[1] = predicted_state[1, 0]

        # ***Task for you*** (0.5 points)
        # Make the Kalman update here with kal_P_init = kal_P, predicted = predicted_state and measured = measured_position
        _, _, kal_x, kal_P = kalman_update(kal_P, predicted_state, measured_position)

        # ***Task for you*** (0.5 points)
        # Where is the estimated position?
        estimated_position = np.zeros(2) # change this! For instance:
        estimated_position[0] = kal_x[0, 0]
        estimated_position[1] = kal_x[0, 1]
        
        center_coordinates_estimate = (int(np.round(estimated_position[1])),
                                        int(np.round(estimated_position[0])))

        cv2.circle(frame_current, center_coordinates_estimate, 10, (255, 0, 0), 2)
        center_coordinates_estimate_label = tuple(map(sum, zip(center_coordinates_estimate, (20,15))))
        cv2.putText(frame_current,                      # image
                'Kalman estimation',                    # text
                center_coordinates_estimate_label,      # start position
                font,                                   # font
                0.5,                                    # size
                (255, 0, 0),                            # BGR color
                1,                                      # thickness
                cv2.LINE_AA)                            # type of line
        '''
        cv2.imshow("Tracking result", curr_frame)
        action = cv2.waitKey(1)
        if action==27:
            break
        
    print("Tracking finished.")
    cv2.destroyAllWindows()
