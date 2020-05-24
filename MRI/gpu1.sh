# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 400 --test 400  --noise 0
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 0 --version 2
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 150 --version 2
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 150 --version 2
# python SAE.py --gpu 0 --cn1 4 --cn2 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 20000 --val 200 --test 200 --noise 80 --version 2 --loss1 correntropy --loss2 correntropy
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 200 --version 2
# python noiseAE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn False --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise_level 10 --us_factor 4 --version 2

## May 15, 2020
# python AE_train.py --gpu 1 --cn 6 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --train 65000 --val 400 --test 1000
python AE_train.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --version 3 --train 65000 --val 400 --test 1000