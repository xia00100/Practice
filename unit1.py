import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
img_path = './demo.jpeg'
# print(os.path.exists(img_path))


def destroy_window():
    if cv2.waitKey():
        cv2.destroyAllWindows()


# img_grey = cv2.imread(img_path, 0)

# cv2.imshow('demo', img_grey)
# destroy_window()

# print(img_grey)
#
# print(img_grey.dtype)
#
# print(img_grey.shape)   # 915*640

# img = cv2.imread(img_path)
# cv2.imshow('img_color', img)
# destroy_window()

# print(img)
# print(img.shape)    # 915 * 640 * 3

# img crop
# img_crop = img[0:600, 100:500]
# cv2.imshow('img_crop', img_crop)
# destroy_window()

# B, G, R = cv2.split(img)
# cv2.imshow('B', B)
# cv2.imshow('G', G)
# cv2.imshow('R', R)
# destroy_window()


# change color
def random_light_color(img, angle):
    B, G, R = cv2.split(img)

    b_rand = random.randint(-angle, angle)
    g_rand = random.randint(-angle, angle)
    r_rand = random.randint(-angle, angle)

    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    return img_merge


# img_random_color = random_light_color(img)
# cv2.imshow('img_random_color', img_random_color)
# destroy_window()

# gamma correction
# img_gh_path = './unit1/guanghua.jpeg'
# print(os.path.exists(img_gh_path))
# img_gh = cv2.imread(img_gh_path)
# cv2.imshow('img_gh', img_gh)
# destroy_window()


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


# img_brighter = adjust_gamma(img_gh, 2)
# cv2.imshow('img_dark', img_gh)
# cv2.imshow('img_brighter', img_brighter)
# destroy_window()


# histogram
# img_small_brighter = cv2.resize(img_brighter,
#                                 (int(img_brighter.shape[0]*0.5),
#                                  int(img_brighter.shape[1]*0.5)))
# plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')
# img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# cv2.imshow('small img', img_small_brighter)
# cv2.imshow('histogram', img_output)
# destroy_window()

# rotation
# M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 45, 1)    # center, angle, scale
# img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow('rotated lenna', img_rotate)
# destroy_window()

# print(M)

# img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow('rotated', img_rotate2)
# destroy_window()


# M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale
# img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow('rotated lenna', img_rotate)
# destroy_window()

# print(M)

# Affine Transform
# rows, cols, ch = img.shape
# pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
# pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
#
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(img, M, (cols, rows))

# cv2.imshow('affine', dst)
# destroy_window()


# perspective transform
def random_warp(img, random_margin=60):
    height, width, channels = img.shape

    # warp:
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))

    return M_warp, img_warp


# if __name__ == '__main__':
#     M_warp, img_warp = random_warp(img)
#     cv2.imshow('lenna_warp', img_warp)
#     destroy_window()
