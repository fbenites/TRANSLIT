#download.sh
#unzip data
run_unzip.sh
#collect data into dictionaries which will be used in merge_datasets.py
python prepare_jrc.py
python prepare_ama_col2018.py
python prepare_geonames.py
python prepare_wiki.py
python clean_wiki_names.py

#merge the data
python merge_datasets.py
