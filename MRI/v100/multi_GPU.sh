# ### Apr. 22, 2020
# python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 1 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-4 --G_lr 1e-4 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 2 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 5e-5 --G_lr 5e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 3 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 5e-4 --G_lr 5e-4 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# 
# python train_DA.py --gpu 4 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.85 --dAcc2 0.95 --iters 200000 --bz 400
# python train_DA.py --gpu 5 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.85 --dAcc2 0.95 --iters 200000 --bz 400

# JOB: python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 1 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-6 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 2 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-7 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 3 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-7 --G_lr 1e-8 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000

## Apr. 23, 2020
#JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 10 --bz 400 --lamda 0.1

#JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.1
#JOB: python train_DA1.py --gpu 1 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.01
#JOB: python train_DA1.py --gpu 2 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.001
#JOB: python train_DA1.py --gpu 3 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

#JOB: python train_DA1.py --gpu 0 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.1
#JOB: python train_DA1.py --gpu 1 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.01
#JOB: python train_DA1.py --gpu 2 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.001
#JOB: python train_DA1.py --gpu 3 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

# JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 10.0
# JOB: python train_DA1.py --gpu 1 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 100.0
# JOB: python train_DA1.py --gpu 2 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0
# JOB: python train_DA1.py --gpu 3 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

## Apr. 26, 2020
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-3 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --lr 1e-6 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 2.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 2.0 --nb_trg_labels 0

## Apr. 27, 2020
# DA + target labels
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0


# JOB: python TF.py --gpu 0 --docker True --lr 1e-6 --iters 10000 --bz 100 --nb_trg_labels 100
# JOB: python TF.py --gpu 1 --docker True --lr 1e-6 --iters 10000 --bz 100 --nb_trg_labels 200
# JOB: python TF.py --gpu 2 --docker True --lr 1e-6 --iters 10000 --bz 100 --nb_trg_labels 300
# JOB: python TF.py --gpu 3 --docker True --lr 1e-6 --iters 10000 --bz 100 --nb_trg_labels 400

# JOB: python TF.py --gpu 0 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 500 --source_scratch True
# JOB: python TF.py --gpu 1 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 1000 --source_scratch True
# JOB: python TF.py --gpu 2 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 2000 --source_scratch True
# JOB: python TF.py --gpu 3 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 4000 --source_scratch True

## Apr. 28, 2020
# parser.add_argument("--gpu", type=int)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--shared", type = str2bool, default = True)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--mmd_param", type = float, default = 1.0)
# parser.add_argument("--trg_clf_param", type = float, default = 1.0)
# parser.add_argument("--src_clf_param", type = float, default = 1.0)
# parser.add_argument("--source_scratch", type = str2bool, default = True)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)
# parser.add_argument("--den_bn", type = str2bool, default = False)
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-7 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 2
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 2
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 2
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 2

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-3 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch False --den_bn False --lr 1e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 1 --clf_v 1
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 1 --clf_v 1
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 1 --clf_v 1
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 1 --clf_v 1


## compton
# python mmd_DA.py --gpu 2 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# python mmd_DA.py --gpu 5 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# parser.add_argument("--gpu", type=int)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--source_scratch", type = str2bool, default = False)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)
# JOB: python TF.py --gpu 0 --docker True --lr 1e-6 --iters 30000 --bz 100 --nb_trg_labels 100
# JOB: python TF.py --gpu 1 --docker True --lr 1e-6 --iters 30000 --bz 100 --nb_trg_labels 200
# JOB: python TF.py --gpu 2 --docker True --lr 1e-6 --iters 30000 --bz 100 --nb_trg_labels 300
# JOB: python TF.py --gpu 3 --docker True --lr 1e-6 --iters 30000 --bz 100 --nb_trg_labels 400

# parser.add_argument("--gpu", type=int)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--source_scratch", type = str2bool, default = False)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)
# parser.add_argument("--DA_FLAG", type = str2bool, default = False)
# parser.add_argument("--source_name", type = str, default = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k')
# parser.add_argument("--DA_name", type = str, default = 'mmd-1.0-lr-0.0001-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.5-sclf-1.0-trg_labels-0')

# JOB: python TF.py --gpu 0 --docker True --lr 1e-6 --iters 30000 --bz 100 --source_scratch False --nb_trg_labels 100 --clf_v 2 --DA_FLAG True --DA_name 'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-trg_labels-0-vclf-2'
# JOB: python TF.py --gpu 1 --docker True --lr 1e-6 --iters 30000 --bz 100 --source_scratch False --nb_trg_labels 200 --clf_v 2 --DA_FLAG True --DA_name 'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-trg_labels-0-vclf-2'
# JOB: python TF.py --gpu 2 --docker True --lr 1e-6 --iters 30000 --bz 100 --source_scratch False --nb_trg_labels 300 --clf_v 2 --DA_FLAG True --DA_name 'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-trg_labels-0-vclf-2'
# JOB: python TF.py --gpu 3 --docker True --lr 1e-6 --iters 30000 --bz 100 --source_scratch False --nb_trg_labels 400 --clf_v 2 --DA_FLAG True --DA_name 'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-trg_labels-0-vclf-2'

# JOB: python TF.py --gpu 0 --docker True --lr 1e-5 --iters 30000 --bz 100 --nb_trg_labels 100 --clf_v 2
# JOB: python TF.py --gpu 1 --docker True --lr 1e-5 --iters 30000 --bz 200 --nb_trg_labels 200 --clf_v 2
# JOB: python TF.py --gpu 2 --docker True --lr 1e-5 --iters 30000 --bz 300 --nb_trg_labels 300 --clf_v 2
# JOB: python TF.py --gpu 3 --docker True --lr 1e-5 --iters 30000 --bz 400 --nb_trg_labels 400 --clf_v 2

# Apri 30, 2020
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'

# JOB: python TF.py --gpu 0 --docker True --lr 5e-6 --iters 50000 --bz 100 --nb_trg_labels 100 --clf_v 1 --dataset 'total'
# JOB: python TF.py --gpu 1 --docker True --lr 5e-6 --iters 50000 --bz 100 --nb_trg_labels 200 --clf_v 1 --dataset 'total'
# JOB: python TF.py --gpu 2 --docker True --lr 5e-6 --iters 50000 --bz 100 --nb_trg_labels 300 --clf_v 1 --dataset 'total'
# JOB: python TF.py --gpu 3 --docker True --lr 5e-6 --iters 50000 --bz 100 --nb_trg_labels 400 --clf_v 1 --dataset 'total'

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-6 --iters 100000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset 'dense'

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset 'dense'

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 100 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 0.5 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 0.5 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 0.5 --clf_v 1 --dataset 'dense'
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 2.0 --trg_clf_param 0.5 --clf_v 1 --dataset 'dense'

# May 1, 2020
# v100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 0 --dis_fc 256 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 0 --dis_fc 256 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 0 --dis_fc 256 --dis_bn True --source_scratch True --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 0 --dis_fc 256 --dis_bn True --source_scratch False --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0

## on compton
# python adda_DA.py --gpu 5 --docker False --dis_cnn 2 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# python adda_DA.py --gpu 6 --docker False --dis_cnn 2 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# python adda_DA.py --gpu 2 --docker False --dis_cnn 2 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0

## on predator
#python adda_DA.py --gpu 0 --docker False --dis_cnn 2 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0

# v100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0

# v100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 400

# v100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 400

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 0.5 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 0.5 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 0.5 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 200000 --bz 400 --dis_param 0.5 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 400


# on gauss
# python adda_DA.py --gpu 1 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn False --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn False --source_scratch True --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn False --source_scratch False --den_bn False --clf_v 2 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0

# parser.add_argument("--gpu_num", type=int)
# parser.add_argument("--nb_cnn", type = int)
# parser.add_argument("--bn", type = str2bool, default = False)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--nb_train", type = int)
# # parser.add_argument("--noise", type = float)
# # parser.add_argument("--sig_rate", type = float)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--optimizer", type = str)
# parser.add_argument("--nb_steps", type = int, default = 100000)
# parser.add_argument("--dataset", type = str, default = 'total')

# on compton
# python TF.py --docker False --gpu 0 --lr 5e-6 --iters 50000 --bz 400 --source_scratch False --dataset total --nb_trg_labels 100
# python train_target.py --docker Flase --gpu_num 0 --nb_cnn 4 --bn False --lr 5e-6 --nb_train 100

# May 2, 2020
# compton 
# python mmd_DA.py --gpu 2 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 70 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --clf_v 1
# python mmd_DA.py --gpu 5 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --clf_v 1
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 70 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1

# v100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 400

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 400

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0--nb_trg_labels 0 --dataset total
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --nb_trg_labels 0 --dataset total
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 100 --dataset total
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 100 --dataset total

## predator
# python mmd_DA.py --gpu 5 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --clf_v 1
# python TF.py --gpu 1 --docker True --lr 1e-6 --iters 50000 --bz 100 --nb_trg_labels 100 --clf_v 1 --dataset 'dense'
# python TF.py --gpu 2 --docker True --lr 1e-6 --iters 50000 --bz 100 --nb_trg_labels 200 --clf_v 1 --dataset 'dense'

## compton
# python adda_DA.py --gpu 2 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 70 --dataset dense
# python adda_DA.py --gpu 2 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 70 --dataset dense
# python adda_DA.py --gpu 5 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 100 --dataset dense
# turing
# python adda_DA.py --gpu 6 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 200 --dataset dense
# deep learning
# python adda_DA.py --gpu 6 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 300 --dataset dense
# python adda_DA.py --gpu 3 --docker False --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 400 --dataset dense

## May 3, 2020
# 100
# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 400

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 100
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 200
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 300
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 400

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 100 --dataset total
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 200 --dataset total
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 300 --dataset total
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 400 --dataset total

# JOB: python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 70 --dataset total
# JOB: python adda_DA.py --gpu 1 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --nb_trg_labels 70 --dataset total
# JOB: python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-4 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --nb_trg_labels 0 --dataset total
# JOB: python adda_DA.py --gpu 3 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 5e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --nb_trg_labels 0 --dataset total

# predator
# mmd-1.0-lr-1e-05-bz-400-iter-100000-scr-None-shar-True-fc-128-bn-False
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total
# python mmd_DA.py --gpu 0 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 1 --docker False --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

# turing
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 500000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 5 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 500000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 500000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 2 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 500000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# 
# python mmd_DA.py --gpu 1 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 500000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.2 --clf_v 1 --dataset total --valid 100


# turing


# gauss
# python mmd_DA.py --gpu 1 --docker False --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100


# May 5, 2020
## v100
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.4 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.1 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 5e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.4 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 5e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 5e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 5e-6 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.1 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.4 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.3 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.1 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.6 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.7 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.8 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.9 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.6 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.7 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.8 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.9 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.6 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.7 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.8 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.9 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100

## predator
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 0.5 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 0.5 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# 
# python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 0.5 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# 
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100

## turing
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.25 --clf_v 1 --dataset total --valid 100

## compton
# python mmd_DA.py --gpu 2 --docker False --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.7 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 5 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 100000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.25 --clf_v 1 --dataset total --valid 100
# 
# python mmd_DA.py --gpu 6 --docker False --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 100000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.7 --clf_v 1 --dataset total --valid 100

## gauss
# python TF.py --gpu 1 --docker False --lr 5e-5 --iters 100000 --bz 100 --nb_trg_labels 1000 --clf_v 1 --dataset total --valid 100

## deeplearning
# python TF.py --gpu 1 --docker False --lr 1e-6 --iters 100000 --bz 100 --nb_trg_labels 1000 --clf_v 1 --dataset total --valid 100




