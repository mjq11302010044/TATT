python3 main.py --arch="tatt" --batch_size=64 --STN --mask --use_distill --gradient  --sr_share --stu_iter=1 --vis_dir='TATT/'  --rotate_train=5 --rotate_test=0 --learning_rate=0.001 --tssim_loss --test_model="ASTER" 
python3 main.py --arch="tatt" --batch_size=64 --STN --mask --use_distill --gradient  --sr_share --stu_iter=1 --vis_dir='TATT_ft/'  --rotate_train=5 --rotate_test=0 --learning_rate=0.0002 --resume="ckpt/TATT/" --tssim_loss --test_model="ASTER"

