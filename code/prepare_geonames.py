import shelve


import pandas as pd

geonames = pd.read_csv("../data/geonames/alternateNamesV2.txt", sep='\t', error_bad_lines=False)

names = {}
for  index,l1 in geonames.iterrows():
    if l1[1] in names:
        names[l1[1]].append([l1[2],l1[3]])
    else:
        names[l1[1]]=[[l1[2],l1[3]]]


skeys=list(set([tk for tk in names for tj in names[tk] if tj[0]=="en"]))

names_clean={}

languages=set()
for key in skeys:
    for data in names[key]:
        if type(data[0]) is not float:
            if len(data[0])==2:

                languages.add(data[0])
                if key not in names_clean:
                    names_clean[key]=[data]
                else:
                    names_clean[key].append(data)

nk=list(names_clean.keys())
for name in nk:
    if len(names_clean[name])==1:
        names_clean.pop(name, None)


nk=list(names_clean.keys())
data = shelve.open("../artefacts/geonames_clean.shf")

for name in nk:
    data[str(name)]=names_clean[name]


data.close()
