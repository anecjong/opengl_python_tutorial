import cv2
import os

class SpecularMap():
    def __init__(self, image_path: str = None) -> None:
        if image_path is None:
            return
        self.image_path = image_path
        
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        ret, self.binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.binary = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
    
    def save_img(self, ) -> None:
        ext = '.' + (self.image_path).split('.')[-1]
        cv2.imwrite(self.image_path.replace(ext, "_otsu"+ext), self.binary)


if __name__ == "__main__":
    spec_map = SpecularMap(os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "..",
                            "..",
                            "resources",
                            "logo.png",
                            ))
    spec_map.save_img()