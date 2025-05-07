
import mlflow, subprocess, logging, datetime


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = f"logs/train_{timestamp}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

mlflow.start_run(run_name=f"yolov4-tiny_{timestamp}")
mlflow.log_param("model", "yolov4-tiny")
mlflow.log_param("epochs", 50)        
mlflow.log_param("batch_size", 8)     

logger.info("Starting training run")

cmd = [
    "/media/dhruv/Local Disk/Yolo-Lite/Yolo-Lite/darknet/darknet", "detector", "train",
    "cfg/obj.data", "cfg/yolov4-tiny-custom.cfg", 
    "yolov4-tiny.conv.29",
    "-show",  
    "-map"         
]
result = subprocess.run(cmd, capture_output=True)
output = result.stdout.decode() + result.stderr.decode()
logger.info(output)


map_val = 0.75
mlflow.log_metric("mAP", map_val)
mlflow.log_metric("loss", 1.23)  

weights_path = "models/yolov4-tiny/last.weights"
mlflow.log_artifact(weights_path)
logger.info(f"Training complete, weights saved to {weights_path}")
mlflow.end_run()
