from flask import Flask, request, jsonify
import os
import logging
from table_detection import TableDetection
from ocr import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def load_initial_setup():
    """
    Function that runs when the server starts
    """
    logger.info("Server is starting up...")

    logger.info("Initializing table detector...")

    global detector
    # performing table detection
    detector = TableDetection(
        model_path="foduucom/table-detection-and-extraction",
    )

    logger.info("Models loaded successfully")


@app.route("/process_image", methods=["POST"])
def process_image():
    try:

        # Access the global detector
        global detector

        # Check if detector is initialized
        if detector is None:
            return (
                jsonify({"error": "Table detector not initialized", "status": "error"}),
                500,
            )

        # Check if the post request has the file path
        if "image_path" not in request.json:
            return jsonify({"error": "No image path provided", "status": "error"}), 400

        base_folder = "./images"
        image_path = os.path.join(base_folder, request.json["image_path"])

        # Validate the file path
        if not os.path.exists(image_path):
            return (
                jsonify({"error": "Image path does not exist", "status": "error"}),
                404,
            )

        # performing OCR
        tables_cropped_images = detector.run(image_path)
        ocr_processor = OCRProcessor()
        tables_data = []
        for table_index, table_image in enumerate(tables_cropped_images):
            ocr_processor.predict(table_image)
            ocr_processor.process_and_draw_boxes()
            ocr_processor.save_image()
            table_rows = ocr_processor.extract_rows()

            table_no = table_index + 1
            rows_data = []
            for row_index, row in enumerate(table_rows):
                row_no = row_index + 1
                rows_data.append({"row_no": row_no, "data": row})

            tables_data.append({"table_no": table_no, "rows": rows_data})
        detector.clear()
        return jsonify(tables_data)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    # Run initial setup
    load_initial_setup()

    # Start the server
    app.run(debug=True, host="0.0.0.0", port=5000)
