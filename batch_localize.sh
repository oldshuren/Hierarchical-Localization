#!/bin/sh
db_descriptor=$1
query_descriptor=$2
query_images=$3
sfm_workspace=$4
db_prefix=${5:-ALL}
query_prefix=${6:-ALL}
num_matched=${7:-50}

output=${sfm_workspace}/localized
pairs=localize_pairs_netvlad${num_matched}

superpoint_features=feats-superpoint-n4096-r1600

mkdir ${output}

echo "Find pairs from query to databse"
python -m hloc.pairs_from_retrieval --descriptors $db_descriptor --query_descriptors $query_descriptor --num_matched $num_matched --output ${output}/${pairs}.txt --db_prefix $db_prefix --query_prefix $query_prefix

echo "Create local features for query"
python -m hloc.extract_features --image_dir $query_images --export_dir ${output} --conf superpoint_inloc

echo "Create match database"

python -m hloc.match_features --export_dir ${sfm_workspace} --output_dir $output --features ${superpoint_features} --query_features ${output}/${superpoint_features}.h5 --pairs ${output}/${pairs}.txt

echo "create query with intrinsics"
for f in $query_images/*.png ; do
    basename=$(basename $f .png)
    echo "${basename}.png SIMPLE_RADIAL 640 384 2167.68 320 192 0.957543" >> ${output}/query_with_intrinsics.txt
done

echo "Localize from sfm"
python -m hloc.localize_sfm --reference_sfm ${sfm_workspace}/sfm_superpoint+superglue/models/0 --queries ${output}/query_with_intrinsics.txt --features ${output}/${superpoint_features}.h5 --matches ${output}/${superpoint_features}_matches-superglue_${pairs}.h5 --retrieval ${output}/${pairs}.txt --results ${output}/localize_from_sfm_results.txt
