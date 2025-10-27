import cv2
cv2.startWindowThread()


# for shape detector, i will need:
# load_image, process_image, find_contours (edges)
class ShapeDetector():
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None


    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f'Failed to load image from path from {self.image_path}')
        cv2.imshow('Original', self.original_image)
        cv2.waitKey(0)
        return self.original_image
    
    def image_to_greyscale(self):
        if self.original_image is None:
            raise FileNotFoundError('Image has not been loaded')
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        cv2.imshow('Greyscale', self.processed_image) 
        cv2.waitKey(0)
        return self.processed_image



    

if __name__ == '__main__':
    image = 'images/shapes.png'
    shape = ShapeDetector(image)
    shape.load_image()
    shape.image_to_greyscale()



