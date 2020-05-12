# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 400 --test 400  --noise 0
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 0 --version 2
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 150 --version 2
# python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 150 --version 2
# python SAE.py --gpu 0 --cn1 4 --cn2 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 20000 --val 200 --test 200 --noise 80 --version 2 --loss1 correntropy --loss2 correntropy
python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn False --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 50 --version 2