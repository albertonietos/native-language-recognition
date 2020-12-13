# %% Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn import preprocessing
import opensmile
from tqdm import tqdm
import csv

# %% Prepare opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
# %% Open metadata files
dataset_path = "-------"  
train_and_dev_file =  "lab/ComParE2016_Nativeness.tsv"
test_file = "lab/ComParE2016_Nativeness-test.tsv"
meta = pd.read_csv(dataset_path + train_and_dev_file, sep="\t")
orig_labels = meta.L1.unique()
meta_test = pd.read_csv(dataset_path + test_file, sep="\t")
label_encoder = preprocessing.LabelEncoder()
meta.L1 = label_encoder.fit_transform(meta.L1)
meta_test.L1 = label_encoder.transform(meta_test.L1)
meta_train = meta[meta['file_name'].str.contains('train')]
meta_devel = meta[meta['file_name'].str.contains('devel')]

# %% print encoding map
transformed_labels = label_encoder.transform(orig_labels)
for i in range(len(transformed_labels)):
    print("{} -> {}".format(orig_labels[i],transformed_labels[i]))




# %% Open output file and write header
output_file = "ComParE_test.csv"
fout = open(output_file,"w")

# %% Generating sets
for index, row in tqdm(meta_test.iterrows(), total=len(meta_test)):
    fname = row["file_name"]
    label = row["L1"]
    df = smile.process_file(dataset_path + "wav/" + fname)
    df.insert(loc=0, column="L1", value=label)
    df.head()
    if index == 0:
        df.to_csv(fout, index=False, header=True, line_terminator='\n')
    else:
        df.to_csv(fout, index=False, header=False, line_terminator='\n')

# %% Save output file
fout.close()
