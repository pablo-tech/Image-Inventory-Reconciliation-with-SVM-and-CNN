
from os import listdir
from os.path import isfile, join

import json

folder_name = "Metadata"

only_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
print only_files

print "LENGTH=", len(only_files)

for file in only_files:
    file_path = folder_name+"/"+file
    with open(file_path) as metadata_file:
        metadata_json = json.load(metadata_file)
        meta_list = metadata_json["BIN_FCSKU_DATA"]
        for meta_key in meta_list:
            meta_data = meta_list[meta_key]
            # print meta_data["quantity"]

# you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
