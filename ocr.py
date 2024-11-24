from paddleocr import PaddleOCR
import cv2

class OCRProcessor:
    def __init__(self, use_angle_cls=True, lang='en'):
        self.use_angle_cls = use_angle_cls
        self.ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=lang)


    def predict(self, image):
        # self.image = cv2.imread(img_path)
        self.image = image
        self.result = self.ocr.ocr(self.image, cls=self.use_angle_cls)
        self.boxes = [line[0] for line in self.result[0]]  # Assuming one table in image
        self.txts = [line[1][0] for line in self.result[0]]
        self.scores = [line[1][1] for line in self.result[0]]

    def draw_box(self, box, color=(0, 255, 0), thickness=2):
        # Convert the box to integer coordinates for OpenCV
        box = [list(map(int, point)) for point in box]

        # Draw lines between each pair of points to create a rectangle/polygon
        for i in range(len(box)):
            start_point = tuple(box[i])
            end_point = tuple(box[(i + 1) % len(box)])  # Connect last point to the first point
            self.image = cv2.line(self.image, start_point, end_point, color, thickness)

        return self.image

    def process_and_draw_boxes(self):
        for box in self.boxes:
            self.draw_box(box)
        return self.image
    
    def save_image(self, output_path="draw.png"):
        """
        Save the image with bounding boxes drawn.

        :param output_path: The path where the output image will be saved.
        """
        cv2.imwrite(output_path, self.image)
    
    def format_rows_data(self):
        ocr_results = []
        for index in range(len(self.boxes)):
            box = self.boxes[index]
            text = self.txts[index]
            ocr_results.append({
                "coords": box,
                "text": text
            })
        return ocr_results

    def extract_rows(self):
        ocr_results = self.format_rows_data()
        # Calculate the average y-coordinate of each bounding box
        for result in ocr_results:
            y_coords = [coord[1] for coord in result['coords']]
            x_coords = [coord[0] for coord in result['coords']]
            result['avg_y'] = sum(y_coords) / len(y_coords)
            result['avg_x'] = sum(x_coords) / len(x_coords)

        # Sort the entire list by the average y-coordinate
        ocr_results.sort(key=lambda x: x['avg_y'])

        # Group texts into rows, and within each row sort by x-coordinate
        rows = []
        current_row = []
        current_y = ocr_results[0]['avg_y']
        threshold = 10  # Define a threshold for considering bounding boxes to be in the same row

        for result in ocr_results:
            if abs(result['avg_y'] - current_y) < threshold:
                current_row.append(result)
            else:
                # Sort the row by the x-coordinate to maintain order
                current_row.sort(key=lambda x: x['avg_x'])
                rows.append([item['text'] for item in current_row])
                current_row = [result]
                current_y = result['avg_y']

        if current_row:
            current_row.sort(key=lambda x: x['avg_x'])
            rows.append([item['text'] for item in current_row])

        return rows
