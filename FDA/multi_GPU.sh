## May 10, 2020
# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 33000 --val 200 --test 200 --version 2 --dataset scattered
# JOB: python AE.py --gpu 1 --cn 5 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 33000 --val 200 --test 200 --version 2 --dataset scattered
# JOB: python AE.py --gpu 2 --cn 6 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 33000 --val 200 --test 200 --version 2 --dataset scattered
# JOB: python AE.py --gpu 3 --cn 7 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 33000 --val 200 --test 200 --version 2 --dataset scattered

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 85000 --val 200 --test 200 --version 2 --dataset total
# JOB: python AE.py --gpu 1 --cn 5 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 85000 --val 200 --test 200 --version 2 --dataset total
# JOB: python AE.py --gpu 2 --cn 6 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 85000 --val 200 --test 200 --version 2 --dataset total
# JOB: python AE.py --gpu 3 --cn 7 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 85000 --val 200 --test 200 --version 2 --dataset total

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 36000 --val 200 --test 200 --version 2 --dataset hetero
# JOB: python AE.py --gpu 1 --cn 5 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 36000 --val 200 --test 200 --version 2 --dataset hetero
# JOB: python AE.py --gpu 2 --cn 6 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 36000 --val 200 --test 200 --version 2 --dataset hetero
# JOB: python AE.py --gpu 3 --cn 7 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 36000 --val 200 --test 200 --version 2 --dataset hetero

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 7100 --val 200 --test 200 --version 2 --dataset dense
# JOB: python AE.py --gpu 1 --cn 5 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 7100 --val 200 --test 200 --version 2 --dataset dense
# JOB: python AE.py --gpu 2 --cn 6 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 7100 --val 200 --test 200 --version 2 --dataset dense
# JOB: python AE.py --gpu 3 --cn 7 --fr 32 --ks 3 --bn True --lr 1e-5 --step 300000 --bz 50 --train 7100 --val 200 --test 200 --version 2 --dataset dense

## May 11, 2020
JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 100 --train 7100 --val 200 --test 200 --version 1 --dataset dense --loss mse
JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 100 --train 36000 --val 200 --test 200 --version 1 --dataset hetero --loss mse
JOB: python AE.py --gpu 2 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 100 --train 33000 --val 200 --test 200 --version 1 --dataset scattered --loss mse
JOB: python AE.py --gpu 3 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 100 --train 9000 --val 200 --test 200 --version 1 --dataset fatty --loss mse