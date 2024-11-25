# Table Detection and OCR Application

This project features table detection and Optical Character Recognition (OCR) using advanced models to detect and extract text from tables in images.

## Models Details

### Table Detection Model
- **Model**: YOLO v8
- **Purpose**: Detects bordered and borderless tables in images.
- **Performance**: Fine-tuned on a vast dataset to achieve high accuracy.
- **Model Source**: [Hugging Face YOLO v8 Model](https://huggingface.co/foduucom/table-detection-and-extraction) 

### OCR Model
- **Model**: PaddleOCR
- **Purpose**: Performs text detection and extraction from detected tables.
- **Model Source**: [Repo link].(https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file) 

## Setup and Execution Instructions

To run the Flask application, follow these steps:

1. **Install Requirements:**
   - Run the following command to install the necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Execute the Server:**
   - Start the Flask server by executing:
     ```bash
     python server.py
     ```

## API Usage Guidelines

To process images with the API after the server is running:

1. **Input Parameters:**
   - Use Postman or a similar tool to send a POST request to the URL: `http://localhost:5000/process_image`.
   - Include a JSON payload with the following structure:
     ```json
     {
         "Image_path": "table3.png"
     }
     ```

   **Note:** Ensure the image is located in the project's `./images` directory. If you need to process a new image, add it to this directory first.

2. **Expected Output:**
   - The API will return a JSON response with the structure:
     ```json
     [
         {
             "rows": [
                 {
                     "data": [],
                     "row_no": 1
                 }
             ],
             "table_no": 1
         }
     ]
     ```

## Demo Run

Here's an example output using the sample image provided in the assignment:

```json
[
    {
        "rows": [
            {
                "data": [
                    "Plastic",
                    "Acetone",
                    "Flame test",
                    "Heat",
                    "Crease color"
                ],
                "row_no": 1
            },
            {
                "data": [
                    "1",
                    "No effect",
                    "Green color",
                    "Softens",
                    "None"
                ],
                "row_no": 2
            },
            {
                "data": [
                    "2",
                    "Softened",
                    "No change",
                    "No change",
                    "White"
                ],
                "row_no": 3
            },
            {
                "data": [
                    "3",
                    "No effect",
                    "Red color",
                    "Softens",
                    "None"
                ],
                "row_no": 4
            },
            {
                "data": [
                    "4",
                    "No effect",
                    "Green color",
                    "Softens",
                    "none"
                ],
                "row_no": 5
            }
        ],
        "table_no": 1
    }
]

## Python version

This code is executed on python 3.11 
