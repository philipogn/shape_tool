import cv2
cv2.startWindowThread()


# for shape detector, i will need:
# load_image, process_image, find_contours (edges)
class ShapeDetector():
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.contours = []


    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f'Failed to load image from path from {self.image_path}')
        return self.original_image
    
    def preprocess_image(self):
        greyed = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150) # find edges
        self.processed_image = edges

        cv2.imshow('Greyscale', edges) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return edges
    
    def find_contours(self):
        _, binary = cv2.threshold(self.processed_image, 225, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.contours = [c for c in contours if cv2.contourArea(c) > 50] # filters out noisy contours

        image = cv2.drawContours(self.original_image, self.contours, -1, (0, 255, 0), 2)
        cv2.imshow('Greyscale', image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.contours
    


    

if __name__ == '__main__':
    # image = 'images/shapes.png'
    # image = 'images/road_sign.jpg'
    image = 'images/coins.jpeg'
    shape = ShapeDetector(image)
    shape.load_image()
    shape.preprocess_image()
    shape.find_contours()



