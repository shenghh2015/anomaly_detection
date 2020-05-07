cd /home/shenghuahe/DA_Observer/CLB_FDA/v100
python2 job_parser.py 'multi_GPU2.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
   sleep 30s &
done
wait
