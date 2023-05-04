# mos_odom

## train
python train.py -d ../dataset -ac config/arch/mos.yml -dc config/data/local-test.yaml -p output/logs/xxxxxx

## infer
python infer.py -d ../dataset -ac config/arch/mos-infer.yml -dc config/data/local-test.yaml -m /home/yu/Resp/Mos_odom_valid_best.pth.tar