cd /home/shenghuahe/anomaly_detectioin/FDA
python2 v100_job_parser.py 'multi_GPU.sh'
for i in $(seq 0 3)
do
   sh v100_jobs/job_$i.sh&
   sleep 10s &
done
wait
