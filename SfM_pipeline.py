from datetime import datetime
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--outputs', type=Path, required=True)
parser.add_argument('--sfm_pairs', type=Path, required=False)


args = parser.parse_args()

images = args.images
outputs = args.outputs

sfm_pairs = args.sfm_pairs

if sfm_pairs is None:
    sfm_pairs = images / 'pairs_netvlad50.txt'
    
sfm_dir = outputs / 'sfm_superpoint+superglue'

feature_conf = extract_features.confs['superpoint_inloc']
matcher_conf = match_features.confs['superglue']

features = feature_conf['output']
feature_file = f"{features}.h5"
match_file = f"{features}_{matcher_conf['output']}_{sfm_pairs.stem}.h5"

# ## Extract local features

print ('start at {}'.format(datetime.now()))
       
extract_features.main(feature_conf, images, outputs)

print ('finished extract_features at {}'.format(datetime.now()))

# ##  Matching
#  Pairs were created using image retrieval and `hloc/pairs_from_retrieval.py`.


match_features.main(matcher_conf, sfm_pairs, features, outputs, exhaustive=False)
print ('finished match_features at {}'.format(datetime.now()))


# ## SfM reconstruction
# Run COLMAP on the features and matches.

# In[ ]:


reconstruction.main(
    sfm_dir,
    images,
    sfm_pairs,
    outputs / feature_file,
    outputs / match_file)

print ('finished reconstruction at {}'.format(datetime.now()))


