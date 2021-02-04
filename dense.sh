#!/bin/sh
image_path=$1
sparse_model=$2
dense_workspace=${3:-$sparse_model/../dense}

colmap image_undistorter --image_path $image_path --input_path $sparse_model --output_path $dense_workspace --output_type COLMAP --max_image_size 2000
colmap patch_match_stereo --workspace_path $dense_workspace --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path $dense_workspace --workspace_format COLMAP --input_type geometric --output_path $dense_workspace/fused.ply
#colmap poisson_mesher --input_path $dense_workspace/fused.ply --output_path $dense_workspace/meshed-poisson.ply
#colmap delaunay_mesher --input_path $dense_workspace --output_path $dense_workspace/meshed-delaunay.ply
