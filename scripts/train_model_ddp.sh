# find all configs in configs/
config=pool_activitynet_64x64_k9l4
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4

# ------------------------ need not change -----------------------------------
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun \
--config_path src/experiment/options/anet/tgn_lgi/LGI.yml \
--method_type tgn_lgi \
--dataset anet \
--num_workers 4 \
--distributed