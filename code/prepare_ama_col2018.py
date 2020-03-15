
path="../data/NETransliteration-COLING2018/"
files=["wd_katakana","wd_japanese","wd_hebrew","wd_korean","wd_russian",
       "wd_chinese","wd_arabic"]

lang=["jp","jp","he","ko","ru","ch","ar"]
names={}
for i1,fname in enumerate(files):
    f1=open(path+fname)
    for line in f1:
        ws=line.strip().split("\t")
        if ws[0] in names:
            if ws[1] not in names[ws[0]]:
                names[ws[0]].append([lang[i1],ws[1]])
        else:
            names[ws[0]]=[[lang[i1],ws[1]]]


import shelve


data = shelve.open("../artefacts/amazon_data.shf")

nck = list(names.keys())

                    
for name in nck:
    data[name]=names[name]

data.close()
