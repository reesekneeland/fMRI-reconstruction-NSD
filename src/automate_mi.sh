export CUDA_VISIBLE_DEVICES="0"
for s in 1; do
    for dr in 0.9; do
        for rep in {0..9}; do
            # for mode in imagery vision; do
            python Reconstructions_mi.py \
                --autoencoder_name autoencoder_subj0${s}_4x_locont_no_reconst \
                --data_path /home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/data \
                --subj $s \
                --model_name subj0${s}_hypatia_${dr}_dropout_mlp \
                --vd_cache_dir ../versatile_diffusion \
                --recons_per_sample 16 \
                --rep $rep  \
                --dropout_rate $dr
                    
            
        done    
    done
done
# python Reconstructions.py \
#     --autoencoder_name=autoencoder_subj05_4x_locont_no_reconst \
#     --data_path=../data \
#     --subj=5 \
#     --model_name=prior_257_final_subj05_bimixco_softclip_byol \
#     --vd_cache_dir=../versatile_diffusion \
#     --recons_per_sample=16

# python Reconstructions.py \
#     --autoencoder_name=autoencoder_subj07_4x_locont_no_reconst \
#     --data_path=../data \
#     --subj=7 \
#     --model_name=prior_257_final_subj07_bimixco_softclip_byol \
#     --vd_cache_dir=../versatile_diffusion \
#     --recons_per_sample=16