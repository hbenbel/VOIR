import cv2


class SquareFinder():
    def __init__(self, image, canny_threshold):
        self.image = cv2.medianBlur(image, 9)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.canny_threshold = 50 if canny_threshold is None else canny_threshold
        self.squares = []

    def FindSquares(self):
        return

    def RemoveBorders(self):
        squares = []
        for square in self.squares:
            if square[0] > 3 and square[1] > 3:
                squares.append(square)
        self.squares = squares

    def DrawSquares(self, frame):
        for square in self.squares:
            cv2.polylines(frame, square, True, (0, 255, 0), 2, cv2.LINE_AA)

    def BinaryClassification(self):
        return not not self.squares