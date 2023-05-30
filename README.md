# mos_odom

## train
python train.py -d ../dataset -ac config/arch/mos.yml -dc config/data/local-test.yaml -p output/logs/xxxxxx

## infer
python infer.py -d ../dataset -ac config/arch/mos-infer.yml -dc config/data/local-test.yaml -m /home/yu/Resp/Mos_odom_valid_best.pth.tar

## viz mos(bin & label)
python scripts/visualize_mos.py -d../dataset -p log -s 8

## evaluate
python scripts/evaluate_mos.py -d ../dataset -p log -s valid

python train_motion_pwc.py -d ../dataset -ac config/arch/mos-motion-pwc.yml -dc config/data/motion.yaml