import cv2
from math import sqrt
import numpy as np


def computeCosine(anchor, pt1, pt2, epsilon = 1e-10):
    x1 = pt1[0] - anchor[0]
    y1 = pt1[1] - anchor[1]
    x2 = pt2[0] - anchor[0]
    y2 = pt2[1] - anchor[1]
    return (x1 * x2 + y1 * y2) / (sqrt((x1**2 + y1**2) * (x2**2 + y2**2)) + epsilon)


class SquareFinder():
    def __init__(self, image, canny_threshold, min_contour_area_size):
        self.image = cv2.medianBlur(image, 9)
        self.canny_threshold = 50 if canny_threshold is None else canny_threshold
        self.min_contour_area_size = 100 if min_contour_area_size is None else min_contour_area_size
        self.squares = []

    def FindSquares(self):
        for channel in range(3):
            image_single_channel = cv2.extractChannel(self.image, channel)
            edges = cv2.Canny(image_single_channel, 5, self.canny_threshold, apertureSize = 5)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                contour = contours[i]
                arc_length = 0.02 * cv2.arcLength(contour, True)
                approximative_contour = cv2.approxPolyDP(contour, arc_length, True)
                max_cosine = 0
                if len(approximative_contour) == 4 and cv2.isContourConvex(approximative_contour) and cv2.contourArea(approximative_contour) > self.min_contour_area_size:
                    for j in range(2, 5):
                        cosine = computeCosine(approximative_contour[j - 1][0], approximative_contour[j % 4][0], approximative_contour[j - 2][0])
                        max_cosine = max(max_cosine, cosine)
                    if max_cosine < 0.3:
                        self.squares.append(approximative_contour)

    def RemoveBorders(self):
        squares = []
        for square in self.squares:
            if square[0][0][0] > 3 and square[0][0][1] > 3:
                squares.append(square)
        self.squares = squares

    def DrawSquares(self, frame):
        for square in self.squares:
            cv2.polylines(frame, square, True, (0, 0, 255), 10)

    def BinaryClassification(self):
        return not not self.squares