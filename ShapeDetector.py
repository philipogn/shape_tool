import cv2
import matplotlib.pyplot as plt
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
        # to detect yellow
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        # define range for yellow in HSV
        lower_yellow = (20, 100, 100)
        upper_yellow = (40, 255, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) # create mask for yellow

        greyed = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0) # apply blur to reduce noise
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150) # find edges
        combined = cv2.bitwise_or(edges, mask_yellow)

        self.processed_image = combined
        # plt.imshow(self.processed_image)
        return edges
    
    def find_contours(self):
        # convert greyscale to binary, nonzero pixel = 1, zero pixel = 0, need to set threshold to distinguish between yellow & white
        # but this makes it unable to detect shapes??
        # _, binary = cv2.threshold(self.processed_image, 255, 255, cv2.THRESH_TOZERO_INV)
        # RETR_EXTERNAL just for detect outline for shape. Not RETR_TREE, too noisy
        # contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contours = [c for c in contours if cv2.contourArea(c) > 100] # filters out noisy contours

        image = cv2.drawContours(self.original_image, contours, -1, (0, 255, 0), 2)
        return self.contours

# measure_contour, what_shape?
class ShapeMeasurer:
    ''' static util methods in class without need of instances '''
    @staticmethod
    def measure_contour(contour: list) -> dict:
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
    def classify_shape(num_sides: int, width: float, height: float) -> str:
        shape_dict = {3: 'Triangle', 4: ('Square', 'Rectangle'), 5: 'Pentagon', 6: 'Hexagon', 7: 'Heptagon', 8: 'Octagon'}
        if num_sides == 4:
            proportion = width / height # check if proportion ~1 for squares
            if 0.95 <= proportion <= 1.05: 
                return shape_dict[4][0]
            else:
                return shape_dict[4][1]
        elif num_sides in shape_dict:
            return shape_dict[num_sides]
        elif num_sides >= 9:
            return 'Circle/Polygon'
        else:
            return 'Unknown shape'
        
class ResultVisualizer:
    def __init__(self, image):
        self.image = image.copy()

    def draw_contours(self, contours, measurements):
        for contour, measure in zip(contours, measurements):
            cv2.drawContours(self.image, [contour], -1, (0, 0, 0), 2)

            # put text of shape info in center
            x, y = measure['x'] + (measure['width'] // 4), measure['y'] + (measure['height'] // 2)
            label = f'{measure["shape_type"]}'
            cv2.putText(
                self.image, 
                label, 
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 0), 
                2,
            )

            area_label = f'area: {measure["area"]}'
            cv2.putText(
                self.image,
                area_label,
                (x, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )
        return self.image
    
    def display(self):
        # convert back to original
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image)
        plt.title('Shape Detection Results')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class ShapeApp:
    def __init__(self, image_path):
        self.image_path = image_path
        self.detector = None
        self.measurements = []

    def run(self):
        self.detector = ShapeDetector(self.image_path)
        self.detector.load_image()
        self.detector.preprocess_image()
        contours = self.detector.find_contours()
        print(f'Found {len(contours)} shapes')

        measurer = ShapeMeasurer()
        for shape in contours:
            self.measurements.append(measurer.measure_contour(shape))
        
        print('Visualizing results..')
        visualizer = ResultVisualizer(self.detector.original_image)
        visualizer.draw_contours(contours, self.measurements)
        visualizer.display()
        


    

if __name__ == '__main__':
    IMAGE_PATH = 'images/shapes.png'
    # IMAGE_PATH = 'images/road_sign.jpg'
    # IMAGE_PATH = 'images/coins.jpeg'
    shape = ShapeApp(IMAGE_PATH)
    shape.run()



