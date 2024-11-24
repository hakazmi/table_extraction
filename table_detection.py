import os
from ultralyticsplus import YOLO
import cv2
import torch


class TableDetection:
    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        agnostic_nms=False,
        max_det=1000,
    ):
        """
        Initializes the TableDetection class with model settings

        :param model_path: Path to the YOLO model.
        :param conf_threshold: NMS confidence threshold.
        :param iou_threshold: NMS IoU threshold.
        :param agnostic_nms: NMS class-agnostic flag.
        :param max_det: Maximum number of detections per image.
        """
        # Load model and set parameters
        self.model = YOLO(model_path)
        self.model.overrides["conf"] = conf_threshold
        self.model.overrides["iou"] = iou_threshold
        self.model.overrides["agnostic_nms"] = agnostic_nms
        self.model.overrides["max_det"] = max_det
        self.table_cropped = []

    def predict(self, image_path):
        """
        Perform inference on the image and return the results.
        :param image_path: Path to the image for detection.
        """
        # Verify if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Load image
        self.image = cv2.imread(image_path)
        results = self.model.predict(self.image)
        return results

    def draw_boxes(self, results):
        """
        Draw bounding boxes on the image.

        :param results: The results returned by the model's predict method.
        """
        for r in results:
            for box in r.boxes:
                coordinates = (box.xyxy).tolist()[0]
                left, top, right, bottom = (
                    int(coordinates[0]),
                    int(coordinates[1]),
                    int(coordinates[2]),
                    int(coordinates[3]),
                )
                cv2.rectangle(self.image, (left, top), (right, bottom), (255, 0, 0), 2)

                cropped_image = self.image[top:bottom, left:right]
                self.table_cropped.append(cropped_image)
                # self.image = cropped_image

    def save_image(self, output_path="out.png"):
        """
        Save the image with bounding boxes drawn.

        :param output_path: The path where the output image will be saved.
        """
        cv2.imwrite(output_path, self.image)

    def run(self, image_path):
        """
        Execute the full process: predicts the bounding boxes and saves the image.
        """
        results = self.predict(image_path)
        self.draw_boxes(results)
        self.save_image()
        return self.table_cropped

    def clear(self):
        self.table_cropped.clear()

    def export_to_coreml(self):
        self.model.export(format="coreml")

    def save_model(self):
        torch.save(self.model, "model.pt")
