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
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5, help='detection confidence threshold')
    return parser.parse_args()


def get_detected_boxes(img, outs, conf_threshold):
    img_height, img_width, img_channels = img.shape
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                det_center_x = int(detection[0] * img_width)
                det_center_y = int(detection[1] * img_height)

                det_width = int(detection[2] * img_width)
                det_height = int(detection[3] * img_height)

                det_corner_x = int(det_center_x - det_width / 2)
                det_corner_y = int(det_center_y - det_height / 2)

                boxes.append([det_corner_x, det_corner_y, det_width, det_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes


def classify_objects(img, conf_threshold):
    net = cv2.dnn.readNet("yolov3_clothes.weights", "yolov3_clothes.cfg")
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = get_detected_boxes(img, outs, conf_threshold)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    with open("clothes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for i in range(len(boxes)):
        if i in indexes:
            label = f"{classes[class_ids[i]]}: {100 * confidences[i]:.2f}%"
            print(label)

    if not boxes:
        print('No object was detected.')


def main():
    args = parse_args()
    image = img_from_url(args.url)
    if image is not None:
        classify_objects(image, args.thresh)


if __name__ == '__main__':
    main()
