# parser.add_argument("--gpu_num", type=int)
# parser.add_argument("--nb_cnn", type = int)
# parser.add_argument("--bn", type = bool)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--nb_train", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--optimizer", type = str)
# parser.add_argument("--nb_steps", type = int, default = 100000)
#JOB: python train_target.py --gpu_num 0 --nb_cnn 4 --bn True --lr 1e-5 --nb_train 85000 --bz 400 --optimizer 'Adam' --nb_steps 10000
#JOB: python train_target.py --gpu_num 1 --nb_cnn 4 --bn True --lr 1e-6 --nb_train 85000 --bz 400 --optimizer 'Adam' --nb_steps 10000
#JOB: python train_target.py --gpu_num 2 --nb_cnn 4 --bn True --lr 1e-7 --nb_train 85000 --bz 400 --optimizer 'Adam' --nb_steps 10000
#JOB: python train_target.py --gpu_num 3 --nb_cnn 4 --bn True --lr 1e-8 --nb_train 85000 --bz 400 --optimizer 'Adam' --nb_steps 10000
# Apr. 27, 2020
JOB: python train_target.py --gpu_num 0 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 100 --bz 100 --optimizer 'Adam' --nb_steps 50000
JOB: python train_target.py --gpu_num 1 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 200 --bz 100 --optimizer 'Adam' --nb_steps 50000
JOB: python train_target.py --gpu_num 2 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 300 --bz 100 --optimizer 'Adam' --nb_steps 50000
JOB: python train_target.py --gpu_num 3 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 400 --bz 100 --optimizer 'Adam' --nb_steps 50000

# JOB: python train_target.py --gpu_num 0 --nb_cnn 4 --bn False --lr 1e-6 --nb_train 500 --bz 400 --optimizer 'Adam' --nb_steps 10000
# JOB: python train_target.py --gpu_num 1 --nb_cnn 4 --bn False --lr 1e-6 --nb_train 1000 --bz 400 --optimizer 'Adam' --nb_steps 10000
# JOB: python train_target.py --gpu_num 2 --nb_cnn 4 --bn False --lr 1e-6 --nb_train 5000 --bz 400 --optimizer 'Adam' --nb_steps 10000
# JOB: python train_target.py --gpu_num 3 --nb_cnn 4 --bn False --lr 1e-6 --nb_train 10000 --bz 400 --optimizer 'Adam' --nb_steps 10000
