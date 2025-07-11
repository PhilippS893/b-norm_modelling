#!/bin/bash

declare -a types=("sexspecific")
declare -a metrics_pds_asso=("tstat_z-pds")
declare -a model_names=("CNorm" "BNorm")
declare -a names=("male" "female")
declare -a timepoints=("2yr" "4yr")

for model in "${model_names[@]}"
do
  for key in "${names[@]}"
  do
    for tpt in "${timepoints[@]}"
    do
      # prctDev
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" prctDev positive_"${tpt}" concat_pos_"${tpt}" 4x1
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" prctDev negative_"${tpt}" concat_neg_"${tpt}" 4x1

      # tstats
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" tstat "${tpt}" concat_"${tpt}" 4x1
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" tstat_thresh "${tpt}" concat_"${tpt}" 4x1

      # EV
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" EV "${tpt}" concat_"${tpt}" 2x2

      # Kurtosis
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" kurtosis "${tpt}" concat_"${tpt}" 2x2

      # Skew
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" skew "${tpt}" concat_"${tpt}" 2x2

      # logp (from Kruskal-Wallis) of PDS
      ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" logp_thresh "${tpt}" concat_"${tpt}" 2x2

    done
    # logp (from Kruskal-Wallis) of PDS
    ./concat_imgs_single.sh ./plots/glasser/"${model}"/"${key}" logp_thresh delta concat_delta 2x2
  done
done

./concat_imgs_single.sh ./plots/glasser lobe_def lobes concat_lobe 4x1