# ### May 8, 2020
# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 0
# JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 5
# JOB: python AE.py --gpu 2 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 5
# JOB: python AE.py --gpu 3 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 10

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 10
# JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 10
# JOB: python AE.py --gpu 2 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 15
# JOB: python AE.py --gpu 3 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 15

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 20
# JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 20
# JOB: python AE.py --gpu 2 --cn 4 --fr 32 --ks 5 --bn True --lr 5e-6 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 25
# JOB: python AE.py --gpu 3 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 100000 --bz 50 --train 65000 --val 400 --test 400 --noise 25

# ### May 9, 2020
# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 40 --version 1
# JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 50 --version 1

# JOB: python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 40 --version 1
# JOB: python AE.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-5 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 50 --version 1

# JOB: python AE.py --gpu 0 --cn 6 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 40 --version 1
# JOB: python AE.py --gpu 1 --cn 6 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 50 --version 1
# JOB: python AE_ssim.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 40 --version 2
# JOB: python AE_ssim.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-3 --step 300000 --bz 100 --train 65000 --val 200 --test 200 --noise 40 --version 2
# JOB: python AE_ssim.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 100 --train 65000 --val 200 --test 200 --noise 30 --version 2 --loss ssim

# JOB: python AE.py --gpu 1 --cn 6 --fr 32 --ks 5 --bn True --lr 1e-4 --step 300000 --bz 50 --train 65000 --val 200 --test 200 --noise 50 --version 1

# May 12, 2020
# JOB: python SAE.py --gpu 0 --cn1 4 --cn2 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 20000 --val 200 --test 200 --noise 80 --version 1 --loss1 mse --loss2 mse

# JOB: python SAE.py --gpu 0 --cn1 4 --cn2 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 20000 --val 200 --test 200 --noise 80 --version 1 --loss1 correntropy --loss2 mse

# JOB: python SAE.py --gpu 0 --cn1 4 --cn2 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 300000 --bz 50 --train 20000 --val 200 --test 200 --noise 80 --version 2 --loss1 mse --loss2 mse

# python noiseAE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise_level 0 --us_factor 4 --version 2
# JOB: python AE_train.py --gpu 1 --cn 5 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae'
# JOB: python AE_train.py --gpu 0 --cn 5 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mse'
# JOB: python AE_train.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae'
# JOB: python AE_train.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae'
# JOB: python AE_train.py --gpu 1 --cn 6 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae'
# JOB: python AE_train.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn False --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae'
# May 31, 2020
# Run 1
# JOB: python AE_labels.py --gpu 0 --cn 6 --fr 32 --ks 5 --bn True --lr 1e-4 --step 100000 --bz 50 --version 1 --train 65000 --val 400 --test 1000 --loss 'mae' --ano_weight 0.1
# JOB: python AE_labels.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 100000 --bz 50 --version 4 --train 65000 --val 400 --test 1000 --loss 'mae' --ano_weight 0.1
# Run 2
JOB: python AE_labels.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-6 --step 100000 --bz 50 --version 1 --train 65000 --val 400 --test 1000 --loss 'mae' --ano_weight 0.05
JOB: python AE_labels.py --gpu 1 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 100000 --bz 50 --version 3 --train 65000 --val 400 --test 1000 --loss 'mae' --ano_weight 0.05