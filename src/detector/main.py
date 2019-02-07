from square_finder import *
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to the image to be analyzed", required=True)
    parser.add_argument("-t", "--threshold", type=int, help="threshold for canny", required=False)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    sf = SquareFinder(image, args.threshold)
    sf.FindSquares()
    sf.RemoveBorders()
    sf.DrawSquares(image)
    contain_squares = sf.BinaryClassification()
    
    folder = './positive' if contain_squares else './negative'
    os.mkdir(folder)
    cv2.imwrite(folder + '/' + args.image.split("/")[-1])
    print("The image was put in the folder ", folder)

if __name__ == "__main__":
    main()