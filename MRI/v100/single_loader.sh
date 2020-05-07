cd /home/shenghuahe/DA_Observer/CLB_FDA/v100
python2 job_parser.py 'single_GPU.sh'
for i in $(seq 0 1)
do
   sh job_folder/job_$i.sh&
   sleep 10s &
done
wait