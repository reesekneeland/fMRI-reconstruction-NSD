# 1 2 prior_257_final_subj0${s}_bimixco_softclip_byol
export CUDA_VISIBLE_DEVICES="3"
for s in 1; do #5 7
    for dr in 0.4; do
        for rep in {0..9}; do
            # for m in 10 5 1 25; do
                # for mode in vision imagery; do
            python Reconstructions_mi_mcd.py \
                --autoencoder_name autoencoder_subj0${s}_4x_locont_no_reconst \
                --data_path /home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/data \
                --subj $s \
                --model_name subj0${s}_hypatia_${dr}_dropout_mlp_projector \
                --vd_cache_dir ../versatile_diffusion \
                --recons_per_sample 16 \
                --rep $rep  \
                --dropout_rate $dr \
                --projector_dropout
                    
            # done
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