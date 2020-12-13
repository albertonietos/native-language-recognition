import os               # for file operations
from tqdm import tqdm   # for progress bar (optional)
import pandas as pd     # simple csv parsing

# Used for generating the label files for hdf5 generation

# this is the dataset type wanted: test,devel,train
dataset_type = "test"

# these are the paths to the wav folder, label file and the output file
data_folder = "native_language/wav/"

if dataset_type == "train":
  output_file = "datapaths/trainpaths.csv"
  label_file = "native_language/lab/ComParE2016_Nativeness.tsv"
  label_index_start = 0

if dataset_type == "devel":
  output_file = "datapaths/develpaths.csv"
  label_file = "native_language/lab/ComParE2016_Nativeness.tsv"
  label_index_start = 3300

if dataset_type == "test":
  output_file = "datapaths/testpaths.csv"
  label_file = "native_language/lab/ComParE2016_Nativeness-test.tsv"
  label_index_start = 0 #Different file

# loading the label file
labels = pd.read_csv(label_file,sep="\t")
#labels.sort_values("file_name",inplace=True,ignore_index=True)

#print(labels)

# opening the output file
f = open(output_file,"w")

# calculate the number of files requipred to be downloaded (for progress bar)
num_files = len([f for f in os.listdir(data_folder) 
     if f.startswith(dataset_type) and os.path.isfile(os.path.join(data_folder, f))])

#print(num_files)

# writing the output header
f.write("file,label_file\n")

label_to_int = {
  'ARA': 0,
  'CHI': 1,
  'FRE': 2,
  'GER': 3,
  'HIN': 4,
  'ITA': 5,
  'JPN': 6,
  'KOR': 7,
  'SPA': 8,
  'TEL': 9,
  'TUR': 10,
}

label_folder = "datapaths/labelfiles/"

# opening progress bar
with tqdm(total=num_files) as pbar:
    # for each file in the specified data_folder

    label_index = label_index_start

    for file_idx, file in enumerate(sorted(os.listdir(data_folder))):
        # if that file in the correct dataset_type
        if( file.startswith(dataset_type) ):
            # get label (L1) and full file path
            label = labels.loc[label_index,"L1"]

            full_path = os.path.abspath(data_folder + file)

            label_path = os.path.abspath(label_folder + str(label_to_int[label]) + ".csv")

            # write to csv
            f.write("{},{}\n".format(full_path, label_path))
            #f.write("{},{}\n".format(full_path, label_to_int[label]))
            # update progress bar

            label_index += 1

            pbar.update(1)

# close file
f.close()