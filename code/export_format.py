


# json
import json

if "names_merged" not in globals():
    import pickle
    names_merged=pickle.load(open("../data/names_merged.pkl","rb"))
names_merged_s = {}
for uui in names_merged:
    names_merged_s[uui.hex] =  list(names_merged[uui])
json.dump(names_merged_s,open("../artefacts/TRANSLIT.json", "w"))
# csv


