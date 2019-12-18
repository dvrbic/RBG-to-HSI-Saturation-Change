import numpy as np
import cv2
import math

img = cv2.imread('mandrill.png', 1)


def change_saturation(img, sat):
    img = (img.astype(float) / 255)
    HSI_img = np.empty((img.shape[0], img.shape[1], 3), float)
    # create empty RGB array
    r = np.empty([img.shape[0], img.shape[1]], dtype=float)
    g = np.empty([img.shape[0], img.shape[1]], dtype=float)
    b = np.empty([img.shape[0], img.shape[1]], dtype=float)
    # create empty HSI array
    H = np.empty([img.shape[0], img.shape[1]], dtype=float)
    S = np.empty([img.shape[0], img.shape[1]], dtype=float)
    I = np.empty([img.shape[0], img.shape[1]], dtype=float)
    # variables for calculations
    theta = np.empty([img.shape[0], img.shape[1], ], dtype=float)
    c = np.empty([img.shape[0], img.shape[1], ], dtype=float)
    n = np.empty([img.shape[0], img.shape[1], ], dtype=float)
    d = np.empty([img.shape[0], img.shape[1], ], dtype=float)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            theta = 0
            r[i, j] = img[i, j][0]
            g[i, j] = img[i, j][1]
            b[i, j] = img[i, j][2]
            n[i, j] = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            d[i, j] = math.sqrt(((r[i, j] - g[i, j]) ** 2) + ((r[i, j] - b[i, j]) * (g[i, j] - b[i, j])))
            theta = n[i, j] / (d[i, j] + 0.000001)
            theta = math.acos(theta)
            c[i, j] = min(r[i, j], g[i, j], b[i, j])

            S[i, j] = 1 - ((3 / (r[i, j] + g[i, j] + b[i, j])) * c[i, j]) + sat
            if (S[i, j] > 1):
                S[i, j] = 1
            elif (S[i, j] < 0):
                S[i, j] = 0
            I[i, j] = (r[i, j] + g[i, j] + b[i, j]) / 3

            if b[i, j] > g[i, j]:
                H[i, j] = (2 * (math.pi) - theta)
            else:
                H[i, j] = theta

    #HSI_img = cv2.merge((H * 255, S * 255, I * 255))
    HSI_img[...,0] = H*255
    HSI_img[...,1] = S*255
    HSI_img[...,2] = I*255

    return HSI_img


sat = float(input("Please enter your value for the saturation between -1 and 1: "))
while ((sat > 1) or (sat < (-1))):
    sat = float(input("Invalid input, please enter your value for the saturation between -1 and 1: "))
HSI_img = change_saturation(img, sat)
cv2.imwrite('HSI_sat_mandrill.png', HSI_img)


def hsi_to_rgb(img):
    img = img.astype(float) / 255
    RGB_img = np.empty((img.shape[0], img.shape[1], 3), float)
    # create empty HSI array
    H = np.empty([img.shape[0], img.shape[1]], dtype=float)
    S = np.empty([img.shape[0], img.shape[1]], dtype=float)
    I = np.empty([img.shape[0], img.shape[1]], dtype=float)
    # create empty RGB array
    r = np.empty([img.shape[0], img.shape[1]], dtype=float)
    g = np.empty([img.shape[0], img.shape[1]], dtype=float)
    b = np.empty([img.shape[0], img.shape[1]], dtype=float)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            H[i, j] = img[i, j][0]
            S[i, j] = img[i, j][1]
            I[i, j] = img[i, j][2]
            if 0 <= H[i, j] < (2 / 3 * math.pi):
                b[i, j] = I[i, j] * (1 - S[i, j])
                r[i, j] = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j])) / (math.cos((math.pi / 3 - H[i, j]))))
                g[i, j] = (3 * I[i, j] - (r[i, j] + b[i, j]))
            elif ((2 / 3 * math.pi) <= H[i, j] < (4 / 3 * math.pi)):
                r[i, j] = I[i, j] * (1 - S[i, j])
                g[i, j] = I[i, j] * (1 + ((S[i, j] * math.cos((H[i, j] - (2 / 3 * math.pi)))) / (
                    math.cos(math.pi / 3 - (H[i, j] - (2 / 3 * math.pi))))))
                b[i, j] = (3 * I[i, j] - (r[i, j] + g[i, j]))
            elif ((4 / 3 * math.pi <= H[i, j] <= 2 * math.pi)):
                g[i, j] = I[i, j] * (1 - S[i, j])
                b[i, j] = I[i, j] * (1 + ((S[i, j] * math.cos((H[i, j] - (4 / 3 * math.pi)))) / (
                    math.cos(math.pi / 3 - (H[i, j] - (4 / 3 * math.pi))))))
                r[i, j] = (3 * I[i, j] - (g[i, j] + b[i, j]))
    #RGB_img = cv2.merge((r * 255, g * 255, b * 255))
    RGB_img[..., 0] = r * 255
    RGB_img[..., 1] = g * 255
    RGB_img[..., 2] = b * 255

    return RGB_img


# RGB_img = hsi_to_rgb(hue_change_img)
RGB_img = hsi_to_rgb(HSI_img)
cv2.imwrite('hSI_To_RGB_mandrill_sat_change.png', RGB_img)