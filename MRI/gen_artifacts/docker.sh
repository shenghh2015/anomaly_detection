chcon -Rt svirt_sandbox_file_t /home/sh38/Anom_Detection/
docker run --gpus all -v /home/sh38/Anom_Detection/:/data -w /data/anomaly_detection/MRI/gen_artifacts -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:latest