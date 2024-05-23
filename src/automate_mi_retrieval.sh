export CUDA_VISIBLE_DEVICE="2"
for s in 1 2 5 7; do
    for mode in imagery vision; do
        # for rep in 1 2 3 4 5 6 7 8 9 10; do
        python Retrievals_mi.py \
            --model_name prior_257_final_subj0${s}_bimixco_softclip_byol \
            --model_name2 prior_1x768_final_subj0${s}_bimixco_softclip_byol \
            --data_path /home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/data \
            --subj $s \
            --mode $mode
            # --rep $rep  \
            
 
        # done    
    done
done