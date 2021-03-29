import cv2
from urllib.request import urlopen
import numpy as np


def img_from_url(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def main():
    url = 'https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,b_rgb:f5f5f5/chhclityponr312bcvdt/aerobill-legacy91-training-hat-1TrTBz.jpg'

    img = img_from_url(url)
    if img is None:
        print('Invalid url - image not found')
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
