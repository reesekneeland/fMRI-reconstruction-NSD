python Reconstructions.py \
    --autoencoder_name=autoencoder_subj01_4x_locont_no_reconst \
    --data_path=../data \
    --subj=1 \
    --model_name=prior_257_final_subj01_bimixco_softclip_byol \
    --vd_cache_dir=../versatile_diffusion \
    --recons_per_sample=16

python Reconstructions.py \
    --autoencoder_name=autoencoder_subj02_4x_locont_no_reconst \
    --data_path=../data \
    --subj=2 \
    --model_name=prior_257_final_subj02_bimixco_softclip_byol \
    --vd_cache_dir=../versatile_diffusion \
    --recons_per_sample=16

python Reconstructions.py \
    --autoencoder_name=autoencoder_subj05_4x_locont_no_reconst \
    --data_path=../data \
    --subj=5 \
    --model_name=prior_257_final_subj05_bimixco_softclip_byol \
    --vd_cache_dir=../versatile_diffusion \
    --recons_per_sample=16

python Reconstructions.py \
    --autoencoder_name=autoencoder_subj07_4x_locont_no_reconst \
    --data_path=../data \
    --subj=7 \
    --model_name=prior_257_final_subj07_bimixco_softclip_byol \
    --vd_cache_dir=../versatile_diffusion \
    --recons_per_sample=16