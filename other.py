
# def create_blank(width, height, rgb_color=(0, 0, 0)):
#     """Create new image(numpy array) filled with certain color in RGB"""
#     # Create black blank image
#     image = np.zeros((height, width, 3), np.uint8)

#     # Since OpenCV uses BGR, convert the color first
#     color = tuple(reversed(rgb_color))
#     # Fill image with color
#     image[:] = color

#     return image

# Create new blank 300x300 red image

# count_color = 0

# w, h = 300, 300

# for color in palette: 
#     test_image = create_blank(w, h, rgb_color=color)
#     cv2.imwrite(f'output/color-{count_color}.jpg', test_image)
#     count_color += 1



# test_image = create_blank(w, h, rgb_color=dominant)
# cv2.imwrite(f'output/color-dominant.jpg', test_image)



# cv2.putText(crop_img, detected_cat,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

# cv2.imshow("crop_img", crop_img)

# cv2.waitKey(0) 



# # Press 'q' to quit
# if cv2.waitKey(1) == ord('q'):
#     break



# # Draw framerate in corner of frame
# # cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

# # All the results have been drawn on the frame, so it's time to display it.
# cv2.imshow('Object detector', frame)

# # Calculate framerate
# t2 = cv2.getTickCount()
# time1 = (t2-t1)/freq
# frame_rate_calc= 1/time1