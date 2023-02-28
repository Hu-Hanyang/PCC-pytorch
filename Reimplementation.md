## 1 Planar
### 1.1 Revise
1. rename all "plane" to "planar" in the ilqr_config.json, ilqr.py and ilqr_utils.py.

### 1.2 Training PCC
python train_pcc.py \
    --env=planar \
    --armotized=False \
    --log_dir=planar_1 \
    --seed=1 \
    --data_size=5000 \
    --noise=0 \
    --batch_size=128 \
    --lam_p=1.0 \
    --lam_c=8.0 \
    --lam_cur=8.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=True

    python train_pcc.py \
    --env=planar \
    --armotized=False \
    --log_dir=planar_2 \
    --seed=3047 \
    --data_size=5000 \
    --noise=0 \
    --batch_size=128 \
    --lam_p=1.0 \
    --lam_c=8.0 \
    --lam_cur=8.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=True

### 1.3 Showing Training Results
tensorboard --logdir=logs/planar/planar_1
tensorboard --logdir=logs/planar/planar_2

### 1.4 Using iLQR
1. python ilqr.py --task=planar --setting_path="result/planar"
2. python ilqr.py --task=planar --setting_path="result/planar" --epoch=5000
#### 1.4.1 questions
1   settings 中的 armotized是什么意思？
    A: 是否使用linear approximation to the Jacobians.

## 2 Pendulum
### 2.1 Training PCC
python train_pcc.py \
    --env=pendulum\
    --armotized=False \
    --log_dir=pendulum_1 \
    --seed=1 \
    --data_size=20000 \
    --noise=0 \
    --batch_size=128 \
    --lam_p=1.0 \
    --lam_c=8.0 \
    --lam_cur=8.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=True

### 2.2 Showing Training Results
tensorboard --logdir=logs/pendulum

### 2.3 Using iLQR
1. python ilqr.py --task=balance --setting_path="result/pendulum"