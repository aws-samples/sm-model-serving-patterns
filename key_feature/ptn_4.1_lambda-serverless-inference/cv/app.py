import json
import cv2
import numpy as np
import base64

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Create random RGB array with number of classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def load_image(name):
    # Open image
    img = cv2.imread(name)
    height, width = img.shape[:2]

    return img, height, width

def load_image_from_base64(img_string):
    # Decode the base64 string into an image
    base_img = base64.b64decode(img_string)
    npimg = np.fromstring(base_img, dtype=np.uint8)
    img = cv2.imdecode(npimg, 1)

    # fetch image height and width
    height, width = img.shape[:2]

    return img, height, width


def infer_image(img, output_layers):
    # pre-processing the image before feeding into the network
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)

    # Feed the pre-processed blob to the network
    net.setInput(blob)

    # Fetch the result
    outs = net.forward(output_layers)

    return outs

def generate_bounding_boxes(outs, height, width, target_confidence):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > target_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    return boxes, confidences, class_ids

def draw_boxes(img, boxes, confidences, class_ids, indexes, colors, classes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    return img

def lambda_handler(event, context):
    image_string = json.loads(event.get('body'))
    img, height, width = load_image_from_base64(image_string['image'])

    outs = infer_image(img, output_layers)
    boxes, confidences, class_ids = generate_bounding_boxes(outs, height, width, 0.5)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    img = draw_boxes(img, boxes, confidences, class_ids, indexes, colors, classes)
    retval, buffer_img = cv2.imencode('.jpg', img)
    image_string = base64.b64encode(buffer_img).decode('utf8')
    payload = {'body': image_string}

    return {
        'statusCode': 200,
        'body': json.dumps(payload),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
