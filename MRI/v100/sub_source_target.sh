#export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast:/scratch/xray_set'
export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G
# export LSB_JOB_REPORT_MAIL=N
bsub -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>70000] rusage[mem=80GB]' -Is -q anastasio-interactive -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" /bin/bash /home/shenghuahe/DA_Observer/CLB_FDA/v100_cluster/run_source_target.sh
# bsub -J 'myArray[1]' -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>70000] rusage[mem=80GB]' -q general -a 'docker(shenghh2020/xray_v100:3.2)' -gpu "num=4" /bin/bash bsub -J 'myArray[1-3]' -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>70000] rusage[mem=80GB]' -q general -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" /bin/bash /home/shenghuahe/DA_Observer/CLB_FDA/v100_cluster/run_source_target.sh
# bsub -J 'myArray[1]' -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>70000] rusage[mem=80GB]' -q general -a 'docker(shenghh2020/xray_v100:3.2)' -gpu "num=4" /bin/bash bsub -J 'myArray[1-3]' -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>70000] rusage[mem=80GB]' -q general -a 'docker(shenghh2020/xray_v100:3.2)' -gpu "num=4" /bin/bash /home/shenghuahe/DA_Observer/CLB_FDA/v100_cluster/run_source_target.sh
#bsub -J 'myArray[1-1]' -n 4 -R 'span[ptile=4] select[mem>40000] rusage[mem=50GB]' -q general -a 'docker(shenghh2020/xray_v100:latest)' -gpu "num=4" -N -u shenghuahe@wustl.edu /bin/bash /home/shenghuahe/run_xray_para.sh
