# for s in 1 2; do
for i in {1..4}; do
    for s in 1 2 5 7; do
        python Reconstructions.py \
            --autoencoder_name autoencoder_subj0${s}_4x_locont_no_reconst \
            --data_path /home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/data \
            --subj $s \
            --model_name prior_257_final_subj0${s}_bimixco_softclip_byol \
            --vd_cache_dir ../versatile_diffusion \
            --recons_per_sample 16 \
            --rep $i 
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