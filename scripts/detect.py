
import cv2, numpy as np, tensorflow as tf, logging, datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = f"logs/detect_{timestamp}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

interpreter = tf.lite.Interpreter(model_path="models/yolov4-tiny/yolov4-tiny.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
logger.info("Starting webcam detection. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    input_data = cv2.resize(frame, (416, 416)) / 255.0
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    cv2.imshow('YOLOv4-Tiny Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.info("Webcam detection ended.")
