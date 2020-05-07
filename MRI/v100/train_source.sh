cd /home/shenghuahe/DA_Observer/CLB_FDA/v100
python2 job_parser.py 'source_jobs.txt'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
#    sleep 60s &
done
wait
python source_source.py
#python source_target.py
