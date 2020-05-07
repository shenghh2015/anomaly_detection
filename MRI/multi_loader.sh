cd /home/shenghuahe/anomaly_detectioin/MRI
python2 job_parser.py 'multi_GPU.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
   sleep 10s &
done
wait
