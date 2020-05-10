### The script is used to run the jobs for xray imaging of the FDA breast phantom
### Date: 04.02.2020
### By: shenghh

### batch command format:
import os
import argparse

data_folder = '/home/sh38/Anom_Detection/'

parser = argparse.ArgumentParser()
parser.add_argument("job_file_name",type=str)
args = parser.parse_args()

job_file_name = args.job_file_name

os.system('chcon -Rt svirt_sandbox_file_t {}'.format(data_folder))
command_str = 'docker run --gpus all -v {0:}:/data -w /data/anomaly_detection/FDA/ -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:latest sh {1:}'.format(data_folder, job_file_name)
print(command_str)
os.system(command_str)


