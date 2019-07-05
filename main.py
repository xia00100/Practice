import cv2
import random
import argparse

from unit1 import random_warp, adjust_gamma, random_light_color


# img_path = './demo.jpeg'
parser = argparse.ArgumentParser(description='data augment')
parser.add_argument('--img_path', dest='img_path', default='./demo.jpeg', type=str)
args = parser.parse_args()


# if __name__ == '__main':
def data_augment(img_path):
    # img_path = args.img_path

    img = cv2.imread(img_path)
    cv2.imshow('img', img)

    img = adjust_gamma(img, random.randint(1,3))
    cv2.imshow('adjust_gamma', img)

    img = random_light_color(img, random.randint(0,180))
    cv2.imshow('random_light', img)

    _, img = random_warp(img, random_margin=random.randint(30,180))
    cv2.imshow('img_adjustGamma_randomLight_warp', img)

    if cv2.waitKey():
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print(args.img_path)
    data_augment(args.img_path)
