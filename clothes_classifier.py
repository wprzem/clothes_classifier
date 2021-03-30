import cv2
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np
import argparse


def img_from_url(url):
    try:
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except HTTPError:
        print('Invalid url - image not found.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', dest='url', required=True, help='url to the image')
    return parser.parse_args()


def classify_clothes(img):
    net = cv2.dnn.readNet("yolov3_clothes.weights", "yolov3_clothes.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    with open("clothes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for i in range(len(boxes)):
        if i in indexes:
            label = f"{classes[class_ids[i]]}: {100 * confidences[i]:.2f}%"
            print(label)

    if not boxes:
        print('No clothing was detected.')


def main():
    args = parse_args()
    image = img_from_url(args.url)
    if image is not None:
        classify_clothes(image)


if __name__ == '__main__':
    main()
