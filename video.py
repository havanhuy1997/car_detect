import os
import sys
import time

import cv2
from tinydb import TinyDB, Query
from tinydb.operations import increment

import mrcnn.model as modellib
from mrcnn import utils, visualize


# Root directory of the project
from samples.coco.coco import CocoConfig
# set the range needed for considering a new bounding boxes as existent
BB_RANGE = 15

# create database db.json
db = TinyDB('db.json')
bb_table = db.table("bounding_boxes")
offset_table = db.table("offsets")

bb_table.truncate()
bb_query = Query()

iteration = 1

# folder to get the weight file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'custom_dataset_limoilou-3.2021-01-15-14-57.h5')

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# config for the mrcnn model
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # COCO dataset has 80 classes + one background class
    BATCH_SIZE = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MAX_INSTANCES = 100

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'car']
count = 0
# use the mp4 provided file for detection
stream = cv2.VideoCapture("parking_detection.mp4")

cv2.namedWindow("window", cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while stream.isOpened():
    print("-----------------Reading frame:", iteration)
    # Capture frame-by-frame
    ret, frame = stream.read()
    if frame is None:
        break
    rgb_image = frame[:, :, ::-1]
    if ret:
        count += 20 # i.e. at 30 fps, this advances about one second
        stream.set(1, count)
    else:
        stream.release()
        print("stream is released")
        break

    # perform detection within a frame
    results = model.detect([rgb_image], verbose=1)
    r = results[0]
    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    print("Detected boxes:", boxes)
    print("Scores:", scores)
    new_boxs = []
    existing_boxs = []
    for box_index, box in enumerate(boxes):
        y1, x1, y2, x2 = (int(value) for value in box)
        bb = bb_table.search(
            ((y1 - BB_RANGE < bb_query.y1) & (y1 + BB_RANGE > bb_query.y1)) &
            ((y2 - BB_RANGE < bb_query.y2) & (y2 + BB_RANGE > bb_query.y2)) &
            ((x1 - BB_RANGE < bb_query.x1) & (x1 + BB_RANGE > bb_query.x1)) &
            ((x2 - BB_RANGE < bb_query.x2) & (x2 + BB_RANGE > bb_query.x2))
        )
        if bb:
            existing_boxs.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "persistence": bb[0]["persistence"] + 1,
                "confidence": int(scores[box_index] * 100)
            })
        else:
            new_boxs.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "persistence": 1, "confidence": int(scores[box_index] * 100),
            })
    bb_table.truncate()

    # Add new bb
    print("New boxs:", new_boxs)
    bb_table.insert_multiple(new_boxs)

    # Update existing bb
    print("Existing boxs:", existing_boxs)
    bb_table.insert_multiple(existing_boxs)

    # Run detection
    start = time.time()
    masked_image = visualize.get_masked_image(rgb_image, boxes, masks, class_ids, class_names, scores)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    for item in bb_table:
        print(item)

    # Display the resulting frame
    cv2.imshow('window', masked_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    iteration = iteration + 1

# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()
