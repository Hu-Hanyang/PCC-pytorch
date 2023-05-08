## 1 Planar
### 1.1 Revise from the original repository
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

### 1.4 Try more and less obstacles

### 1.5 

### 1.4 Using iLQR
python ilqr.py --task=planar --setting_path="result/planar"
python ilqr.py --task=planar --setting_path="result/planar" --epoch=5000
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
2. Change the balance.json file:
originial:
"start_min": [0, 0],
"start_max": [0, 0],
after changing:
"start_min": [-0.5236, -0.5236],
"start_max": [0.5236, 0.5236],
python ilqr.py --task=balance --setting_path="result/pendulum"

### 2.4 Try to use the trained model in the gym environment


## 3 Cartpole (original)
### 3.1 Training PCC
python train_pcc.py \
    --env=cartpole\
    --armotized=False \
    --log_dir=cartpole_1 \
    --seed=1 \
    --data_size=5000 \
    --noise=0.1 \
    --batch_size=1 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

python train_pcc.py \
    --env=cartpole\
    --armotized=False \
    --log_dir=cartpole_test \
    --seed=1 \
    --data_size=1000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

python train_pcc.py \
    --env=cartpole\
    --armotized=False \
    --log_dir=cartpole_2 \
    --seed=1 \
    --data_size=15000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

debugging:
python train_pcc.py \
    --env=cartpole\
    --armotized=False \
    --log_dir=cartpole_debug \
    --seed=1 \
    --data_size=10 \
    --noise=0.1 \
    --batch_size=2 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

### 3.2 Showing Training Results
tensorboard --logdir=logs/cartpole/cartpole_test
tensorboard --logdir=logs/cartpole/cartpole_2

### 3.3 Using iLQR
python ilqr.py --task=balance --setting_path="result/cartpole" --epoch=5000

## 4 CCartpole (our baseline)
### 4.1 Training PCC
python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole_test \
    --seed=1 \
    --data_size=100 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole_test2 \
    --seed=1 \
    --data_size=200 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

debug lines:
{
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train_pcc.py",
            "args": [
                "--env=ccartpole",
                "--armotized=False",
                "--log_dir=ccartpole_test",
                "--seed=1",
                "--data_size=100"
            ],
            "console": "integratedTerminal",
            "justMyCode": true
        }

python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole1 \
    --seed=1 \
    --data_size=15000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

After revise the bug of image sampling:
python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole_test \
    --seed=1 \
    --data_size=100 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=500 \
    --iter_save=100 \
    --save_map=False

python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole1 \
    --seed=1 \
    --data_size=15000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole_test2 \
    --seed=1 \
    --data_size=200 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=500 \
    --iter_save=100 \
    --save_map=False

python train_pcc.py \
    --env=ccartpole\
    --armotized=False \
    --log_dir=ccartpole_test3 \
    --seed=1 \
    --data_size=10000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False

python train_pcc.py \
    --env=ccartpole\
    --armotized=True \
    --log_dir=ccartpole_debug \
    --seed=1 \
    --data_size=10000 \
    --noise=0.1 \
    --batch_size=32 \
    --lam_p=1.0 \
    --lam_c=7.0 \
    --lam_cur=1.0 \
    --vae_coeff=0.01 \
    --determ_coeff=0.3 \
    --lr=0.0005 \
    --decay=0.001 \
    --num_iter=5000 \
    --iter_save=1000 \
    --save_map=False
### 3.2 Showing Training Results
tensorboard --logdir=logs/ccartpole

### 3.3 Using iLQR
python ilqr_comparison.py --task=ccartpole --setting_path="result/ccartpole"
Notice: comment out the code line 147 and line 149 in the /localhome/hha160/anaconda3/envs/pcc/lib/python3.8/site-packages/dmc2gym/wrappers.py.