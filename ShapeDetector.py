from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
cv2.startWindowThread()

class DetectionStrategy(ABC):
    @abstractmethod
    def preprocess(self, image):
        pass

class YellowEdgeDetection(DetectionStrategy): # only good for white backgrounds, or distinguishable bg colour from shape
    def preprocess(self, image):
        # defining range for yellow in HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = (20, 100, 100)
        upper_yellow = (40, 255, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) # create mask for yellow

        greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0) # apply blur to reduce noise
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150) # find edges

        combined = cv2.bitwise_or(edges, mask_yellow) # for including yellow
        plt.title('yellow')
        plt.imshow(combined)
        return combined

class EdgeDetection(DetectionStrategy): 
    def preprocess(self, image):
        greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0) # apply blur to reduce noise
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150) # find edges
        plt.title('norm')
        plt.imshow(edges)
        return edges
    
class AdaptiveThreshold(DetectionStrategy): # adapts to varying brightness levels of image for edge detection, great overall
    def preprocess(self, image):
        greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0)
        # THRESH_BINARY_INV for white shapes on black bg
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        plt.title('adaptive')
        plt.imshow(adaptive)
        return adaptive
    
class Morphology(DetectionStrategy): # method works great for noisy shapes, e.g., coins or shapes with designs..
    ''' WORK IN PROGRESS '''
    def preprocess(self, image):
        greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(greyed, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        plt.title('binary')
        plt.imshow(closing)
        return closing



# load_image, process_image, find_contours (edges)
class ShapeDetector():
    def __init__(self, image_path, strategy: DetectionStrategy):
        self.image_path = image_path
        self.strategy = strategy
        self.original_image = None
        self.processed_image = None
        self.contours = []

    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f'Failed to load image from path from {self.image_path}')
        return self.original_image
    
    def preprocess_image(self):
        self.processed_image = self.strategy.preprocess(self.original_image)
        # plt.imshow(self.processed_image)
        return self.processed_image
    
    def find_contours(self):
        # RETR_EXTERNAL just for detect outline for shape. Not RETR_TREE, too noisy
        # TREE works well with AdaptiveThreshold but returns nested contours, external ignores the shapes and marks the border of the image
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > 100] # filters out noisy contours
        return self.contours

# measure_contour, what_shape?
class ShapeMeasurer:
    ''' static util methods in class without need of instances '''
    @staticmethod
    def measure_contour(contour: list) -> dict:
        area = cv2.contourArea(contour) # calculate area
        perimeter = cv2.arcLength(contour, True) # perimeter
        x, y, w, h = cv2.boundingRect(contour) # this might break or be inaccurate for diamonds, or tilted oblongs? it doesn't so far

        eps = 0.01 * perimeter # greatest distance from contour to approximation contour, measurement of accuracy, low for strictness
        approx = cv2.approxPolyDP(contour, eps, True) # apparently this works for finding sides of a 'shape' according to stackoverflow
        num_sides = len(approx)
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0 # formula to check roundness to detect circles
        proportion = w / h

        shape_type = ShapeMeasurer.classify_shape(num_sides, w, h, approx, circularity)
        return {'area': area, 'perimeter': perimeter, 'width': w, 'height': h, 'x': x, 'y': y, 
                'num_sides': num_sides, 'shape_type': shape_type, 'circularity': circularity, 'proportion': proportion}

    @staticmethod
    def classify_shape(num_sides: int, width: float, height: float, approx, circularity) -> str:
        shape_dict = {3: 'Triangle', 4: ('Square', 'Rectangle'), 5: 'Pentagon', 6: 'Hexagon', 7: 'Heptagon', 8: 'Octagon'}
        proportion = width / height # get aspect ratio proportion
        # threshold ~0.8-0.9, can be unreliable due to jaggedness or blur of contours + checking proportion incase for stretched circles
        if circularity > 0.85 and 0.8 < proportion < 1.2: 
            return 'Circle'
        if num_sides == 4:
            if 0.95 <= proportion <= 1.05: 
                return shape_dict[4][0]
            else:
                return shape_dict[4][1]
        # elif num_sides == 8: # worked but sometimes circles can have somehow have sides ranging 4-11????
        #     if cv2.isContourConvex(approx): # circles show 8 sides, check if convex for circle
        #         return 'Circle/Polygon'
        #     else: 
        #         return shape_dict[8]
        elif num_sides in shape_dict:
            return shape_dict[num_sides]
        else:
            return 'Unknown shape'
        



class ResultVisualizer:
    def __init__(self, image):
        self.image = image.copy()

    def draw_contours(self, contours, measurements):
        for contour, measure in zip(contours, measurements):
            cv2.drawContours(self.image, [contour], -1, (100, 255, 50), 2)

            # put text of shape info in center
            x, y = measure['x'] + (measure['width'] // 5), measure['y'] + (measure['height'] // 3)
            label = f'{measure["shape_type"]}, {measure["num_sides"]} sides'
            cv2.putText(self.image, label, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 50), 1)
            
            area_label = f'Area: {measure["area"]}'
            cv2.putText(self.image, area_label, (x, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 50), 1)
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
        self.strategies = {'1': YellowEdgeDetection(), '2': EdgeDetection(), '3': AdaptiveThreshold(), '4': Morphology()}

    def run(self, strategy_choice: str = '1'):
        strategy = self.strategies.get(strategy_choice, YellowEdgeDetection()) # get choice or run default
        self.detector = ShapeDetector(self.image_path, strategy)
        
        print('Loading and processing image...')
        self.detector.load_image()
        self.detector.preprocess_image()
        contours = self.detector.find_contours()
        print(f'Found {len(contours)} shapes')

        print('Measuring shapes...')
        measurer = ShapeMeasurer()
        for shape in contours:
            self.measurements.append(measurer.measure_contour(shape))
        
        print('Visualizing results...')
        visualizer = ResultVisualizer(self.detector.original_image)
        visualizer.draw_contours(contours, self.measurements)
        visualizer.display()

    def print_information(self):
        for i, measure in enumerate(self.measurements, 1):
            print(f'\nShape {i}')
            print(f'Type: {measure["shape_type"]}, Circularity: {measure["circularity"]:.2f}, AR: {measure["proportion"]}')
            print(f'Area: {measure["area"]:.2f} pixels squared')
            print(f'Perimeter: {measure["perimeter"]:.2f} pixels')
            print(f'Dimensions: {measure["width"]} x {measure["height"]} pixels')



if __name__ == '__main__':
    IMAGE_PATH = 'images/more_shapes.png'
    # IMAGE_PATH = 'images/book.jpg'
    # IMAGE_PATH = 'images/sign_night.jpg'
    # IMAGE_PATH = 'images/traffic_noisy.jpeg'

    # IMAGE_PATH = 'images/coins.jpeg'
    shape = ShapeApp(IMAGE_PATH)
    shape.run('3')
    shape.print_information()
