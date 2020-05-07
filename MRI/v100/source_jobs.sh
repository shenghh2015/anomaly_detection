#parser.add_argument("--gpu_num", type=int)
#parser.add_argument("--nb_cnn", type = int)
#parser.add_argument("--bn", type = str2bool, default = False)
#parser.add_argument("--lr", type = float)
#parser.add_argument("--nb_train", type = int)
#parser.add_argument("--noise", type = float)
#parser.add_argument("--sig_rate", type = float)
#parser.add_argument("--bz", type = int)
#parser.add_argument("--optimizer", type = str)
#parser.add_argument("--nb_steps", type = int, default = 100000)

# Apr. 27, 2020
# python train_source.py --gpu_num 0 --nb_cnn 4 --bn True --lr 5e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
JOB: python train_source.py --gpu_num 0 --nb_cnn 4 --bn False --lr 5e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
JOB: python train_source.py --gpu_num 1 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
JOB: python train_source.py --gpu_num 2 --nb_cnn 6 --bn False --lr 5e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
JOB: python train_source.py --gpu_num 3 --nb_cnn 6 --bn False --lr 1e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
