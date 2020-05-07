cd /home/shenghuahe/DA_Observer/CLB_FDA/v100
python2 job_parser.py 'target_jobs.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
	sleep 10s &
done
wait
python target_target.py
