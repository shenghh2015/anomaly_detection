<<<<<<< HEAD
chcon -Rt svirt_sandbox_file_t /shared2/Data_FDA_Breast/Observer
docker run --gpus 0 -v /shared2/Data_FDA_Breast/Observer:/data -w /data/DA_Observer/CLB_FDA/predator/ -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:latest sh job_gpu_0.sh 
=======
chcon -Rt svirt_sandbox_file_t /home/sh38/Anom_Detection/
docker run --gpus all -v /home/sh38/Anom_Detection/:/data -w /data/anomaly_detection/MRI/ -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:latest
>>>>>>> 62241cfdf148994cfca71f083ee966d54c7a82b0
