import cv2
cv2.startWindowThread()

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
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0) # apply blue to reduce noise
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150) # find edges
        self.processed_image = edges
        # cv2.imshow('Greyscale', edges) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return edges
    
    def find_contours(self):
        # convert greyscale to binary, nonzero pixel = 1, zero pixel = 0, need to set threshold to distinguish between yellow & white
        # but this makes it unable to detect shapes??
        _, binary = cv2.threshold(self.processed_image, 255, 255, cv2.THRESH_TOZERO_INV)
        # RETR_EXTERNAL just for detect outline for shape. Not RETR_TREE, too noisy
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contours = [c for c in contours if cv2.contourArea(c) > 100] # filters out noisy contours

        image = cv2.drawContours(self.original_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Greyscale', image) 
        # cv2.imshow('Greyscale', self.original_image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.contours

# measure_contour, what_shape?
class ShapeMeasurer:
    ''' static util methods in class without need of instances '''
    @staticmethod
    def measure_contour(contour):
        area = cv2.contourArea(contour) # calculate area
        perimeter = cv2.arcLength(contour, True) # perimeter

        x, y, w, h = cv2.boundingRect(contour) # this might break or be inaccurate for diamonds, or tilted oblongs?

        eps = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, eps, True) # apparently this works for finding sides of a 'shape' according to stackoverflow
        num_sides = len(approx)

        shape_type = ShapeMeasurer.classify_shape(num_sides, w, h)

        return {'area': area, 'perimeter': perimeter, 'width': w, 'height': h, 'x': x, 'y': y, 
                'num_sides': num_sides, 'shape_type': shape_type}

    @staticmethod
    def classify_shape(num_sides, width, height):
        if num_sides == 3:
            return 'Triangle'
        elif num_sides == 4:
            proportion = width / height # check if proportion ~1 for squares
            if 0.95 <= proportion <= 1.05: 
                return 'Square'
            else:
                return 'Rectangle'
        elif num_sides >= 4:
            return 'Circle'
        else:
            return 'Unknown shape'

class ShapeApp:
    def __init__(self, image_path):
        self.image_path = image_path
        self.detector = None

    def run(self):
        self.detector = ShapeDetector(self.image_path)
        self.detector.load_image()
        self.detector.preprocess_image()
        contours = self.detector.find_contours()

        print(f'Found {len(contours)} shapes')

        for shape in contours:
            pass

    

if __name__ == '__main__':
    IMAGE_PATH = 'images/shapes.png'
    # IMAGE_PATH = 'images/road_sign.jpg'
    # IMAGE_PATH = 'images/coins.jpeg'
    shape = ShapeApp(IMAGE_PATH)
    shape.run()



