# sRA-Net
official code for paper "Learning Structured Relation Embeddings for
Fine-Grained Fashion Attribute Recognition"

# running environment
python==3.8
pytorch==1.8.0


# Prepare dataset
please download the deepfashion dataset and iFashion-Attribute dataset from Official website. And Change the corresponding path in src/const.py 

# To train the model please run
python -m src.df_sranet_train --conf src.conf.df_sranet_cf

