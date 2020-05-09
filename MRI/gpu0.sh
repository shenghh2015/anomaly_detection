# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu", type=int, default = 1)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--cn", type=int, default = 6)
# parser.add_argument("--fr", type=int, default = 32)
# parser.add_argument("--ks", type=int, default = 5)
# parser.add_argument("--bn", type=str2bool, default = True)
# parser.add_argument("--skp", type=str2bool, default = False)
# parser.add_argument("--res", type=str2bool, default = False)
# parser.add_argument("--lr", type=float, default = 1e-3)
# parser.add_argument("--step", type=int, default = 1000)
# parser.add_argument("--bz", type=int, default = 200)
# parser.add_argument("--dataset", type=str, default = 'total')
# parser.add_argument("--train", type=int, default = 100000)

# python AE.py --gpu 0 --cn 4 --fr 32 --ks 5 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 400 --test 400 --noise 10
python AE.py --gpu 0 --cn 4 --fr 32 --ks 3 --bn True --lr 1e-4 --step 200000 --bz 50 --train 65000 --val 200 --test 200 --noise 40 --version 2
