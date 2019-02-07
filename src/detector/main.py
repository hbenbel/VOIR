from square_finder import *
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to the image to be analyzed", required=True)
    parser.add_argument("-ct", "--canny_threshold", type=int, help="threshold for canny", required=False)
    parser.add_argument("-mcas", "--min_contour_area_size", type=int, help="threshold to determine the minimal square size", required=False)
    args = parser.parse_args()

    print("Loading Image ...")
    image = cv2.imread(args.image)
    print("Image Loaded !")
    sf = SquareFinder(image, args.canny_threshold, args.min_contour_area_size)
    print("Searching for square shaped objects ..")
    sf.FindSquares()
    print("Search completed !")
    sf.RemoveBorders()
    sf.DrawSquares(image)
    contain_squares = sf.BinaryClassification()
    
    folder = './positive' if contain_squares else './negative'
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(folder + '/' + args.image.split("/")[-1], image)
    print("The image was put in the folder ", folder)

if __name__ == "__main__":
    main()