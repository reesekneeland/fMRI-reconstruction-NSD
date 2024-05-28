export CUDA_VISIBLE_DEVICES="2"
jupyter nbconvert Train_MindEye_UMAP.ipynb --to python

metric=correlation
n_neighbors=100
min_dist=0.1
dimensions=1024

python Train_MindEye_UMAP.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_UMAP_upscaled_${dimensions}dim_${metric}_${n_neighbors}neighbors_${min_dist}dist" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \
                        --reduce_dim \
                        --dimensions $dimensions \
                        --metric $metric \
                        --n_neighbors $n_neighbors \
                        --min_dist $min_dist

metric=euclidean
n_neighbors=100
min_dist=0.1
dimensions=1024

python Train_MindEye_UMAP.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_UMAP_${dimensions}dim_${metric}_${n_neighbors}neighbors_${min_dist}dist" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \
                        --reduce_dim \
                        --dimensions $dimensions \
                        --metric $metric \
                        --n_neighbors $n_neighbors \
                        --min_dist $min_dist

metric=haversine
n_neighbors=100
min_dist=0.1
dimensions=1024

python Train_MindEye_UMAP.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_UMAP_${dimensions}dim_${metric}_${n_neighbors}neighbors_${min_dist}dist" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \
                        --reduce_dim \
                        --dimensions $dimensions \
                        --metric $metric \
                        --n_neighbors $n_neighbors \
                        --min_dist $min_dist

metric=cosine
n_neighbors=100
min_dist=0.0
dimensions=1024

python Train_MindEye_UMAP.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_UMAP_${dimensions}dim_${metric}_${n_neighbors}neighbors_${min_dist}dist" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \
                        --reduce_dim \
                        --dimensions $dimensions \
                        --metric $metric \
                        --n_neighbors $n_neighbors \
                        --min_dist $min_dist

metric=cosine
n_neighbors=100
min_dist=0.5
dimensions=1024

python Train_MindEye_UMAP.py --data_path ../data/ \
                        --vd_cache_dir ../versatile_diffusion \
                        --model_name "subj01_hypatia_UMAP_${dimensions}dim_${metric}_${n_neighbors}neighbors_${min_dist}dist" \
                        --subj 1 \
                        --batch_size 32 \
                        --wandb_log \
                        --reduce_dim \
                        --dimensions $dimensions \
                        --metric $metric \
                        --n_neighbors $n_neighbors \
                        --min_dist $min_dist
