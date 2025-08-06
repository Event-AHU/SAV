# train
CUDA_VISIBLE_DEVICES=1 setsid python -m torch.distributed.launch --nproc_per_node=1 train.py --config_file configs/VeRi/vit_base.yml MODEL.DIST_TRAIN True &>/DATA/wuwentao/TransReID-main/output/imagenet_pretrain_2007.log&
# test
setsid python test.py --config_file configs/VeRi/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')" &>/DATA/wuwentao/TransReID-main/output/imagenet_pretrain_2002.log&
/data1/Datasets/wwt/pretrain_data/lunkuo/lunkuo_pkuvd1_part1.zip
--config_file configs/VeRi/vit_adan.yml

CUDA_VISIBLE_DEVICES=5 setsid python -m torch.distributed.launch --master_port='29503' --nproc_per_node=1 train.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DIST_TRAIN True &>/data1/Code/wuwentao/TransReid/log_new/vehiclemaev3_mask75_100W_004_50ep_24_0921.log&

