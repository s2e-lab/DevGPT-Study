# ...

frames = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Aplicăm recunoașterea facială pe fiecare cadru
    frame = frame[:, :, ::-1]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)

    for top, right, bottom, left in faces:
        roi_gray = gray_frame[top:bottom, left:right]
        color = (255, 0, 0)
        stroke = 2

        # Convertim frame-ul într-un obiect cv::Mat
        frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_cv = cv2.UMat(frame_cv)  # Convertim în cv::UMat

        # Aplicăm funcția cv2.rectangle() pe frame-ul convertit
        cv2.rectangle(frame_cv, (left, top), (right, bottom), color, stroke)

        # Convertim frame-ul înapoi la np.ndarray
        frame_cv = frame_cv.get()

        # Continuăm cu restul prelucrărilor și salvăm frame-ul
        # ...

    # ...
