import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import edge_detection as edge # Handles the detection of lane lines
import laneLib as lane
import matplotlib.pyplot as plt # Used for plotting and error checking

filename = 'Detección de carriles/highway_-_10364 (720p).mp4'

# Load a video
cap = cv2.VideoCapture(filename)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

file_size = (1920,1080) # Assumes 1920x1080 mp4
scale_ratio = 0.5 # Option to scale to fraction of original size. 

# Process the video
while cap.isOpened():

    # Capture one frame at a time
    success, frame = cap.read() 
        
    # Do we have a video frame? If true, proceed.
    if success:
        
        # Resize the frame
        width = int(frame.shape[1] * scale_ratio)
        height = int(frame.shape[0] * scale_ratio)
        frame = cv2.resize(frame, (width, height))
            
        # Store the original frame
        original_frame = frame.copy()

        # Create a Lane object
        lane_obj = lane.Lane(orig_frame=original_frame)

        # Perform thresholding to isolate lane lines
        lane_line_markings = lane_obj.get_line_markings()

        # Plot the region of interest on the image
        lane_obj.plot_roi(plot=True)

        # Perform the perspective transform to generate a bird's eye view
        # If Plot == True, show image with new region of interest
        warped_frame = lane_obj.perspective_transform(plot=False)

        # Generate the image histogram to serve as a starting point
        # for finding lane line pixels
        histogram = lane_obj.calculate_histogram(plot=False)	

        # Find lane line pixels using the sliding window method 
        left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
        plot=False)

        # Fill in the lane line
        lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)

        # Overlay lines on the original frame
        frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

        # Calculate lane line curvature (left and right lane lines)
        lane_obj.calculate_curvature(print_to_terminal=False)

        # Calculate center offset  																
        lane_obj.calculate_car_position(print_to_terminal=False)

        # Display curvature and center offset on image
        frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=False)
                
        # Write the frame to the output video file
        #result.write(frame_with_lane_lines2)
            
        # Display the frame 
        cv2.imshow("Frame", frame_with_lane_lines2)

        cv2.imshow("Cenital", lane_obj.warped_frame)

        # Display frame for X milliseconds and check if q key is pressed
        # q == quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    # No more video frames left
    else:
        break
        
# Stop when the video is finished
cap.release()

# Close all windows
cv2.destroyAllWindows() 
