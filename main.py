import cv2
import numpy as np
import time
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readline()]
outputlayers = net.getUnconnectedOutLayersNames()

colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 50
    height, width, channels = frame.shape

# Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)

# Showing infomations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[0] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + "" + str(round(confidence, 2)),
                        (x, y + 30), font, 1, color, 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.imshow("python-life", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
