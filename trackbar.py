import cv2

def nothing(x):
	print ("What is " + str(x))
	pass

if __name__ == "__main__":
	cv2.namedWindow("hsv")
	cv2.createTrackbar('H', 'hsv', 0, 180, nothing)
	cv2.createTrackbar('S', 'hsv', 0, 255, nothing)
	cv2.createTrackbar('V', 'hsv', 0, 255, nothing)
	
	while(1):
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
		# h = cv2.getTrackbarPos('H', 'hsv')
		# s = cv2.getTrackbarPos('S', 'hsv')
		# v = cv2.getTrackbarPos('V', 'hsv')
		# print("{}, {}, {}".format(h, s, v))
