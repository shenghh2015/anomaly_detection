# ### May 9, 2020
JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 0
JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 5
JOB: python AE.py --gpu 2 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 5
JOB: python AE.py --gpu 3 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 10