# export CUDA_VISIBLE_DEVICES="0"
# jupyter nbconvert Train_MindEye.ipynb --to python
# python Train_MindEye.py --data_path ../data/ \
#                         --vd_cache_dir ../versatile_diffusion \
#                         --model_name "subj01_hypatia_default" \
#                         --subj 1 \
#                         --batch_size 32 \
#                         --wandb_log

export CUDA_VISIBLE_DEVICES="0"
jupyter nbconvert Train_MindEye.ipynb --to python
# python Train_MindEye.py --data_path ../data/ \
#                         --vd_cache_dir ../versatile_diffusion \
#                         --model_name "subj01_hypatia_0.15_dropout_mlp_projector" \
#                         --subj 1 \
#                         --dropout_rate 0.15 \
#                         --batch_size 32 \
#                         --wandb_log \
#                         --projector_dropout


python Train_MindEye.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_TEST" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \

