#export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast:/scratch/xray_set'
export LSF_DOCKER_VOLUMES='/home/shenghuahe/anomaly_data:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G
bsub -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>200000] rusage[mem=200GB]' -q anastasio -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" -o /home/shenghuahe/anomaly_data/logs/AD_4GPU_$RANDOM /bin/bash /home/shenghuahe/anomaly_detectioin/MRI/multi_loader.sh