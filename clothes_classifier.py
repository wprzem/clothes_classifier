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
        print('Invalid url - image not found')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', dest='url', required=True, help='url to the image')
    args = parser.parse_args()
    
    img = img_from_url(args.url)
    if img is not None:
        cv2.imshow("Image", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
