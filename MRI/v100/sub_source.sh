#export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast:/scratch/xray_set'
export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G

# export LSB_JOB_REPORT_MAIL=N
bsub -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>200] rusage[mem=200GB]' -q anastasio -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" -o /scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer/logs/$RANDOM /bin/bash /home/shenghuahe/DA_Observer/CLB_FDA/v100/train_source.sh
