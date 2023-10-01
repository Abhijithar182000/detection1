import cv2

# Load YOLO (You Only Look Once) model and its weights
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO names (used for labeling the detected objects)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load an image
image = cv2.imread('image.jpg')

# Get image dimensions
height, width, _ = image.shape

# Create a blob from the image (used to preprocess the image for YOLO)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the neural network
net.setInput(blob)

# Get the output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through the network
outputs = net.forward(output_layer_names)

# Initialize lists to store detected object information
boxes = []
confidences = []
class_ids = []

# Minimum confidence threshold for object detection
confidence_threshold = 0.5

# Iterate through each output layer
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            # Scale the bounding box coordinates back to the original image size
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove redundant bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for the bounding box

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detected objects
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
