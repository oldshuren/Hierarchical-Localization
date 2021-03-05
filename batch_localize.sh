#!/bin/sh
db_descriptor=
query_descriptor=
query_images=
sfm_workspace=
db_prefix=ALL
query_prefix=ALL
num_matched=50
camera_intrinsics="SIMPLE_RADIAL 640 360 658.503 320 180 0.0565491"
extractor=
matcher=
feature_output=
matcher_output=
matcher_batch="1"
output=
usage()
{
      echo "Usage: batch_localize [ -d | --db_descriptor db_descriptor ] [ -q | --query_descriptors query_descriptor ]
                        [ -i | --query_images query_images ] [ --db_prefix db_prefix] [ --query_prefix query_prefix ]
			[ -c | --camera_intrinsics camera_intrinsics ] [ -o| --output output directory ]
			[ --extractor extractor ] [ --matcher matcher ] [ --feature_output feature_output] [ --matcher_output match output] [--matcher_batch match batch size ]
                        [ -s | --sfm_workspace sfm_workspace ] [ -n | --num_matched num_matched ]"
      exit 2
}

PARSED_ARGUMENTS=$(getopt -a -n batch_localize -o d:q:i:s:n:c:o: --long db_descriptor:,query_desciptor:,query_images:,sfm_workspace:,db_prefix:,query_prefix:,num_matched:,camera_intrinsics:,output:,extractor:,matcher:,feature_output:,matcher_output:,matcher_batch: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
    case "$1" in
	-d | --db_descriptor)   db_descriptor="$2"     ; shift 2  ;;
	-q | --query_descriptor)   query_descriptor="$2"     ; shift 2  ;;
	-i | --query_images)   query_images="$2"     ; shift 2  ;;
	--db_prefix)    db_prefix="$2"      ; shift 2  ;;
	--query_prefix)    query_prefix="$2"      ; shift 2  ;;
	-s | --sfm_workspace)   sfm_workspace="$2"     ; shift 2  ;;
	-n | --num_matched)   num_matched="$2"     ; shift 2  ;;
	-c | --camera_intrinsics)   camera_intrinsics="$2"     ; shift 2  ;;
	--extractor)   extractor="$2"     ; shift 2  ;;
	--matcher)   matcher="$2"     ; shift 2  ;;
	--feature_output)   feature_output="$2"     ; shift 2  ;;
	--matcher_output)   matcher_output="$2"     ; shift 2  ;;
	--matcher_batch)   matcher_batch="$2"     ; shift 2  ;;
	# -- means the end of the arguments; drop this, and break out of the while loop
	--) shift; break ;;
	# If invalid options were passed, then getopt should have reported an error,
	# which we checked as VALID_ARGUMENTS when getopt was called...
	*) echo "Unexpected option: $1 - this should not happen."
	   usage ;;
    esac
done

if [ -z "$output" ]; then
    output=${sfm_workspace}/localized
fi
if [ -z "$extractor" ]; then
    extractor=superpoint_inloc
fi
if [ -z "$matcher" ]; then
    matcher=superglue
fi

pairs=localize_pairs_netvlad${num_matched}

if [ -z "$feature_output" ]; then
    feature_output=feats-superpoint-n4096-r1600
fi
if [ -z "$matcher_output" ]; then
    matcher_output=superglue
fi

mkdir ${output}

echo "Find pairs from query to databse"
python -m hloc.pairs_from_retrieval --descriptors $db_descriptor --query_descriptors $query_descriptor --num_matched $num_matched --output ${output}/${pairs}.txt --db_prefix $db_prefix --query_prefix $query_prefix

echo "Create local features for query"
python -m hloc.extract_features --image_dir $query_images --export_dir ${output} --conf $extractor

echo "Create match database"

if [ "$matcher_batch" != "1" ]; then
    echo "Doing multiple pairs matching"
    python -m hloc.match_features_batch --batch $matcher_batch --export_dir ${sfm_workspace} --output_dir $output --features ${feature_output} --query_features ${output}/${feature_output}.h5 --pairs ${output}/${pairs}.txt --conf $matcher
else
    echo "Doing single pairs matching"
    python -m hloc.match_features --export_dir ${sfm_workspace} --output_dir $output --features ${feature_output} --query_features ${output}/${feature_output}.h5 --pairs ${output}/${pairs}.txt --conf $matcher
fi
echo "create query with intrinsics"
for f in $query_images/*.png ; do
    basename=$(basename $f .png)
    echo "${basename}.png $camera_intrinsics" >> ${output}/query_with_intrinsics.txt
done

echo "Localize from sfm"
python -m hloc.localize_sfm --reference_sfm ${sfm_workspace}/sfm_${extractor}+${matcher}/models/0 --queries ${output}/query_with_intrinsics.txt --features ${output}/${feature_output}.h5 --matches ${output}/${feature_output}_matches-${matcher_output}_${pairs}.h5 --retrieval ${output}/${pairs}.txt --results ${output}/localize_from_sfm_results.txt
