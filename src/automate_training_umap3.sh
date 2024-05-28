export CUDA_VISIBLE_DEVICES="3"
jupyter nbconvert Train_MindEye_UMAP.ipynb --to python

metric=cosine
min_dist=0.1
dimensions=1024
for n_neighbors in 3 25 250 1000 5000 10 100 500 2500; do
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
done



