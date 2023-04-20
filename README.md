# mos_odom

## train
python train.py -d ../dataset -ac config/arch/mos.yml -dc config/data/local-test.yaml -p output/logs/xxxxxx

## infer
python infer.py -d ../dataset -m data/model_salsanext_residual_1 -l log -s valid