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

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres, debug=False):
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
        self.debug = debug

        # Load class names for completeness, though pose demo does not use them
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

        # Generate a color palette for potential box drawing
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Will be set after inspecting model outputs
        self.normalized = False

    def draw_detections(self, img, box, score, class_id):
        """Draw a bounding box and label for a detected object."""

        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f'{self.classes[class_id]}: {score:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _make_anchors(self):
        """Build anchor points and corresponding strides for decoding."""
        anchors, strides = [], []
        # Default YOLOv8 strides
        for s in (8, 16, 32):
            ny, nx = self.input_height // s, self.input_width // s
            sy, sx = np.meshgrid(np.arange(ny) + 0.5, np.arange(nx) + 0.5, indexing='ij')
            anchors.append(np.stack((sx, sy), -1).reshape(-1, 2))
            strides.append(np.full((ny * nx, 1), s, dtype=np.float32))
        self.anchor_points = np.concatenate(anchors).astype(np.float32)
        self.stride_tensor = np.concatenate(strides).astype(np.float32)

    def decode_anchor(self, row, anchor_point, stride):
        """Decode a single 56-value model output row using YOLOv8's rules.

        ``self.normalized`` is determined from the raw model outputs. If the
        model already emits values in [0, 1], a second sigmoid would push every
        coordinate toward 1.0 and collapse keypoints. In that case we skip the
        sigmoid here.
        """

        if not self.normalized:
            row = 1 / (1 + np.exp(-row))

        # Decode box center and size
        xy = (row[:2] * 2 - 0.5 + anchor_point) * stride
        wh = (row[2:4] * 2) ** 2 * stride
        score = row[4]

        # Decode keypoints relative to the decoded box center
        kpts = row[5:].reshape(-1, 3)
        kpt_xy = (kpts[:, :2] * 2 - 0.5) * stride + xy
        keypoints = [
            (
                kx * self.img_width / self.input_width,
                ky * self.img_height / self.input_height,
                kc,
            )
            for (kx, ky), kc in zip(kpt_xy, kpts[:, 2])
        ]

        # Convert to top-left corner xywh and scale to image size
        x1y1 = xy - wh / 2
        box = (
            x1y1[0] * self.img_width / self.input_width,
            x1y1[1] * self.img_height / self.input_height,
            wh[0] * self.img_width / self.input_width,
            wh[1] * self.img_height / self.input_height,
        )
        return box, score, keypoints

    def draw_pose(self, img, keypoints, kpt_threshold=0.5):
        """Draw keypoints and skeleton on an image."""

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        # Draw keypoints
        for x, y, c in keypoints:
            if c > kpt_threshold:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw skeleton lines
        for i, j in skeleton:
            x1, y1, c1 = keypoints[i - 1]
            x2, y2, c2 = keypoints[j - 1]
            if c1 > kpt_threshold and c2 > kpt_threshold:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

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
        """Decode model output and draw poses on the image."""

        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        boxes, scores, kpts = [], [], []
        for i in range(rows):
            stride = float(self.stride_tensor[i])
            box, score, keypoints = self.decode_anchor(outputs[i], self.anchor_points[i], stride)
            if score >= self.confidence_thres:
                boxes.append(list(box))
                scores.append(float(score))
                kpts.append(keypoints)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for idx in np.array(indices).flatten():
            self.draw_pose(input_image, kpts[idx])

        return input_image

    def main(self):
        """
        Run inference using an ONNX model and return the image with poses drawn.

        Returns:
            output_img: The output image with drawn poses.
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

        # Prepare anchors for decoding
        self._make_anchors()

        # Inspect raw outputs to determine whether sigmoid has already been applied
        raw = np.transpose(np.squeeze(outputs[0]))
        self.normalized = bool(np.all((raw >= 0) & (raw <= 1)))

        # Optionally decode and print a few raw output rows for debugging
        if self.debug:
            print('Raw model output sample:', raw[0][:8])
            print('Outputs normalized:', self.normalized)
            print('Decoded sample anchors:')
            for i in range(min(3, raw.shape[0])):
                stride = float(self.stride_tensor[i])
                box, score, kpts = self.decode_anchor(raw[i], self.anchor_points[i], stride)
                print({'box': box, 'score': float(score), 'keypoints': kpts[:2]})

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
    parser.add_argument('--debug', action='store_true', help='Print decoded output for a few anchors')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')

    # Create an instance of the Yolov8 class with the specified arguments
    detection = Yolov8(args.model, args.img, args.conf_thres, args.iou_thres, debug=args.debug)

    # Perform pose estimation and obtain the output image
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
