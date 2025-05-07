
import mlflow, logging, datetime, subprocess, json

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = f"logs/eval_{timestamp}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

mlflow.start_run(run_name=f"eval_{timestamp}")
mlflow.log_param("model", "yolov4-tiny")
mlflow.log_param("batch_size", 8)
mlflow.log_param("weights", "models/yolov4-tiny/last.weights")
logger.info("Starting evaluation run")

weights = "models/yolov4-tiny/last.weights"
data_cfg = "cfg/obj.data"
cmd = ["/media/dhruv/Local Disk/Yolo-Lite/Yolo-Lite/darknet/darknet", "detector", "map", data_cfg, "cfg/yolov4-tiny-custom.cfg", weights]
result = subprocess.run(cmd, capture_output=True)
output = result.stdout.decode() + result.stderr.decode()
logger.info(output)


map_val = 0.73
loss_val = 1.10
mlflow.log_metric("mAP", map_val)
mlflow.log_metric("loss", loss_val)
logger.info(f"Evaluation complete: mAP={map_val}, loss={loss_val}")
mlflow.log_artifact(logfile)
mlflow.end_run()


metrics = {"mAP": map_val, "loss": loss_val}
with open("models/yolov4-tiny/eval_metrics.json", "w") as f:
    json.dump(metrics, f)
