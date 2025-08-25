import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

# Allow running the example without installing the package by manually
# adding the repository root to the Python path. This lets users execute the
# script directly with ``python examples/...`` without ``export PYTHONPATH``.
FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml


class Yolov8:

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image and ensure channel count matches the model
        if self.input_channels == 1:
            self.img = cv2.imread(self.input_image, cv2.IMREAD_GRAYSCALE)
            self.img_height, self.img_width = self.img.shape[:2]
            img = cv2.resize(self.img, (self.input_width, self.input_height))
            img = img[:, :, None]  # restore channel axis
        else:
            self.img = cv2.imread(self.input_image)
            self.img_height, self.img_width = self.img.shape[:2]
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize to 0-1 and convert to channel-first layout
        image_data = img.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # (C,H,W)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract predictions and draw them on the
        input image. Supports both object detection and pose estimation models.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Determine if the model is a pose model. Pose models have 5 box/objectness
        # values followed by 3 values for each keypoint (x, y, confidence).
        num_cols = outputs.shape[1]
        is_pose = (num_cols - 5) % 3 == 0 and num_cols < 70

        rows = outputs.shape[0]

        # Calculate the scaling factors for the coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        if is_pose:
            num_kpts = (num_cols - 5) // 3
            boxes, scores, kpts = [], [], []

            for i in range(rows):
                score = outputs[i][4]
                if score >= self.confidence_thres:
                    x, y, w, h = outputs[i][:4]
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    keypoints = outputs[i][5:].reshape(num_kpts, 3)
                    keypoints[:, 0] *= x_factor
                    keypoints[:, 1] *= y_factor

                    boxes.append([left, top, width, height])
                    scores.append(float(score))
                    kpts.append(keypoints)

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

            # Pose models usually predict only "person"
            self.classes = ['person']
            self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

            for idx in np.array(indices).flatten():
                box = boxes[idx]
                score = scores[idx]
                keypoints = kpts[idx]
                self.draw_detections(input_image, box, score, 0)
                self.draw_keypoints(input_image, keypoints)

            return input_image

        # Detection model path
        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][:4]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for idx in np.array(indices).flatten():
            box = boxes[idx]
            score = scores[idx]
            class_id = class_ids[idx]
            self.draw_detections(input_image, box, score, class_id)

        return input_image

    # Pose drawing ---------------------------------------------------------
    def draw_keypoints(self, img, kpts):
        """Draws keypoints and skeleton on the image."""
        skeleton = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
            (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
        ]

        for x, y, conf in kpts:
            if conf > self.confidence_thres:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

        for a, b in skeleton:
            if kpts[a][2] > self.confidence_thres and kpts[b][2] > self.confidence_thres:
                cv2.line(
                    img,
                    (int(kpts[a][0]), int(kpts[a][1])),
                    (int(kpts[b][0]), int(kpts[b][1])),
                    (0, 255, 0),
                    2,
                )

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model. Try CUDA if available,
        # but gracefully fall back to CPU when the required GPU libraries are missing.
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        try:
            session = ort.InferenceSession(self.onnx_model, providers=providers)
        except Exception as e:  # e.g., missing CUDA libraries
            print(f"CUDAExecutionProvider unavailable: {e}. Falling back to CPU.")
            session = ort.InferenceSession(self.onnx_model, providers=['CPUExecutionProvider'])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use. The last two dimensions are
        # width and height regardless of whether the model expects 4D [N,C,H,W]
        # or 5D [N,?,C,H,W] tensors (e.g., some pose models).
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[-1]
        self.input_height = input_shape[-2]
        ch = input_shape[-3] if len(input_shape) >= 3 else None
        self.input_channels = ch if isinstance(ch, int) and ch > 0 else 3

        # Preprocess the image data
        img_data = self.preprocess()

        # If the model expects a 5D tensor, insert the extra dimension
        if len(input_shape) == 5:
            img_data = np.expand_dims(img_data, axis=1)

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        output_img = self.postprocess(self.img, outputs)

        # Return the resulting output image
        return output_img


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n.onnx', help='Input your ONNX model.')
    parser.add_argument('--img', type=str, default=str(ROOT / 'assets/bus.jpg'), help='Path to input image.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--save', type=str, default='', help='Optional path to save the output image')
    parser.add_argument('--show', action='store_true', help='Display the output image in a window')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')

    # Create an instance of the Yolov8 class with the specified arguments
    detection = Yolov8(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    output_image = detection.main()

    # Optionally save the output image
    if args.save:
        cv2.imwrite(args.save, output_image)
        print(f"Saved output image to {args.save}")

    # Optionally display the output image (requires GUI support)
    if args.show:
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Output', output_image)
        cv2.waitKey(0)
