gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thinned_img = thinning(gray_img)
