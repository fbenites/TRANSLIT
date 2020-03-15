import uuid
import shelve
from joblib import Parallel, delayed
import time


def search_cand(candidates, dicts, i1):
    siblings_merged = []
    f1 = open(str(i1)+".log","w")
    print(candidates[0])
    f1.write("# "+str( i1)+" "+ str(len(candidates))+"\n")

    stime = 10
    while "geonames" not in locals():
        try:
            geonames_s = shelve.open("../artefacts/geonames_clean.shf")
            geonames = dict([(tk,geonames_s[tk]) for tk in geonames_s]) 
            geonames_s.close()

        except:
            print("error")
            time.sleep(stime)
            print("continue 1 "+str(i1))
            continue
        #get amazon names
        # https://github.com/steveash/NETransliteration-COLING2018

    while "amazon_names" not in locals():
        try:

            amazon_names_s = shelve.open("../artefacts/amazon_data.shf")
            amazon_names = dict([(tk,amazon_names_s[tk]) for tk in amazon_names_s])
            amazon_names_s.close()
            
        except:
            print("error")
            time.sleep(stime)
            print("continue 2 "+str(i1))

            continue

    #get names from JRC
    #prepare_jrc.py

    while "jrc_names" not in locals():
        try:

            jrc_names_s = shelve.open("../artefacts/jrc_names_new.shf")
            jrc_names = dict([(tk,jrc_names_s[tk]) for tk in jrc_names_s])
            jrc_names_s.close()
        except:
            print("error")
            time.sleep(stime)
            print("continue 3 "+str(i1))

            continue




    #get names from wikipedia

    while "wiki_names_cleaned" not in locals():
        try:

            wiki_names_cleaned_s = shelve.open("../artefacts/wiki_names_cleaned.shf")
            wiki_names_cleaned = dict([(tk,wiki_names_cleaned_s[tk]) for tk in wiki_names_cleaned_s])
            wiki_names_cleaned_s.close()
        except:
            print("error")
            time.sleep(stime)
            print("continue 4 "+str(i1))

            continue

    #get names from google arabic transliteration


    lines = open("../data/google/ar2en.txt").readlines()
    ar2en = {}
    for l1 in lines:
        ws = l1.split("\t")
        #conform to format
        ar2en[ws[1]] = [["ar",ws[0]]]

    dicts_with_no_names_keys=[jrc_names,geonames]
    print("starting "+str(i1))
    found=0
    for c1,cands in enumerate(candidates):
        siblings=[]
        f1.flush()
        f1.write("C1"+str(c1)+"\n")
        for cand1 in cands:
            for d1,nd_r in enumerate(dicts[c1]):
                exec("global nd; nd ="+nd_r)
                if cand1[1] in nd.keys():
                        siblings+=nd[cand1[1]]

                f1.write(".")
                for key in nd:
                            for item in nd[key]:
                                if item[1].find(cand1[1])>-1:
                                    siblings+=nd[key]
                                    found+=1
                                    if nd not in dicts_with_no_names_keys:
                                        siblings+=[key]
                                    break
                                if cand1[1].find(item[1])>-1:
                                    siblings+=nd[key]
                                    found+=1
                                    if nd not in dicts_with_no_names_keys:
                                        siblings+=[key]
                                    break
                                
        siblings_merged.append(siblings)
    if found>1:
            #print("found multiple",found,candidates)
        pass
        #if found==1:
            #    break
    f1.close()
    return siblings_merged





def generate_candidates_long(geonames, amazon_names, jrc_names, wiki_names_clean, ar2en):
    geonames_not_used=[tk for tk in geonames]
    amazon_names_not_used=[tk for tk in amazon_names]
    jrc_not_used=[tk for tk in jrc_names]
    wiki_names_cleaned_not_used = [tk for tk in wiki_names_cleaned]
    ar2en_not_used = [tk for tk in ar2en]


    candidates = []
    candidates_dicts = []

    for candidate in amazon_names_not_used:
        candidates+=[[["en",candidate]]+amazon_names[candidate]]
        candidates_dicts+=[["geonames","jrc_names","wiki_names_cleaned","ar2en"]]

    for     candidate in jrc_not_used:
        candidates+=[jrc_names[candidate]]
        candidates_dicts+=[["geonames","amazon_names","wiki_names_cleaned","ar2en"]]

    for   candidate in geonames_not_used:
        candidates+=[geonames[candidate]]
        candidates_dicts+=[["amazon_names","jrc_names","wiki_names_cleaned","ar2en"]]

        
    for candidate in wiki_names_cleaned_not_used:
        candidates+=[[["en",candidate]]+wiki_names_cleaned[candidate]]
        candidates_dicts+=[["geonames","amazon_names","jrc_names","ar2en"]]

    for candidate in ar2en_not_used:
        candidates+=[[["en",candidate]]+ar2en[candidate]]
        candidates_dicts+=[["geonames","amazon_names","jrc_names","wiki_names_cleaned"]]

    return candidates,candidates_dicts







if "loaded" not in globals():
    #get geonames with geonames_parse.py

    print("loading dicts")

    geonames_s = shelve.open("../artefacts/geonames_clean.shf")
    geonames = dict([(tk,geonames_s[tk]) for tk in geonames_s]) 
    geonames_s.close()
    #get amazon names
    # https://github.com/steveash/NETransliteration-COLING2018

    amazon_names_s = shelve.open("../artefacts/amazon_data.shf")
    amazon_names = dict([(tk,amazon_names_s[tk]) for tk in amazon_names_s])
    amazon_names_s.close()
    #get names from JRC
    #prepare_jrc.py

    jrc_names_s = shelve.open("../artefacts/jrc_names_new.shf")
    jrc_names = dict([(tk,jrc_names_s[tk]) for tk in jrc_names_s])
    jrc_names_s.close()


    #get names from wikipedia

    wiki_names_cleaned_s = shelve.open("../artefacts/wiki_names_cleaned.shf")
    wiki_names_cleaned = dict([(tk,wiki_names_cleaned_s[tk]) for tk in wiki_names_cleaned_s])
    wiki_names_cleaned_s.close()
    #get names from google

    lines = open("../data/google/ar2en.txt").readlines()
    ar2en = {}
    for l1 in lines:
        ws = l1.split("\t")
        #conform to format
        ar2en[ws[1]] = [["ar",ws[0]]]

    loaded = True

#generate unique identifiers
# UID, english text, transliteration list [[lang, transliteration],...]

#keys
#ar2en key english
#wiki_names key english
# amazon_names key english
#jrc key not english






#check how many items







#clean candidates???

# go through the dicts
if "candidates" not in globals():
    print("generating cands") 
    candidates, cand_dicts = generate_candidates_long(geonames, amazon_names, jrc_names, wiki_names_cleaned, ar2en)
print("searching identical", len(candidates))
slices=100000


if "items" not in globals():
    items={}
    dicts_with_no_names_keys=[jrc_names,geonames]
    for d1 in [geonames, amazon_names, jrc_names, wiki_names_cleaned, ar2en]:

        for keyword in d1:
            if d1 not in dicts_with_no_names_keys:
                if keyword not in items:
                    items[keyword]=[[keyword,d1]]
                else:               
                    items[keyword].append([keyword,d1])
            for item in d1[keyword]:
                if item[1] in ["no", 'Kaiserliche Marine', "His Majesty's", 'kaominina']:
                    continue
                if item[1] not in items:
                    items[item[1]]=[[keyword,d1]]
                else:

                    items[item[1]].append([keyword,d1])


#consolidate
merged={}
keyword_uuid={}
consolidated = {}

# set seed for uuid
import random
rd = random.Random()
rd.seed(1337)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))


for item in items:
    group=set()
    for el in items[item]:
        keyword,d1 = el
        if keyword not in keyword_uuid:
            id1=uuid.uuid4()
            while id1 in merged:
                id1=uuid.uuid4()

            keyword_uuid[keyword]=id1
        for sub in d1[keyword]:
            group.add("_".join(sub))
    id1=keyword_uuid[keyword]
    merged[id1]=group

import pickle
pickle.dump(merged,open("../artefacts/names_merged.pkl","wb")) 



# statistics for paper

create_paper_artefacts = False
if create_paper_artefacts:
    dist=[len(tj[0]) for tk in merged.values() for tj in tk]


    from collections import Counter
    # Counter(dist)


    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots() 
    plt.xscale("log")
    n, bins, patches = ax.hist(dist, 600, log=True)
    #ax.semilogx(bins[:-1], n, '--')
    ax.set_xlabel('Name Char Length')
    ax.set_ylabel('Counts')
    ax.set_title(r'Histogram of Names with a given character length')

    # Tweak spacing to prevent clipping of ylabel                                                                                                                                                                                                 

    fig.tight_layout()
    plt.savefig("dist.pdf", format="pdf")


    large_d=[tj.split("_")[1] for tk in merged.values() for tj in tk if len(tj.split("_")[1])>100]
    dist_lang=[tj.split("_")[0] for tk in merged.values() for tj in tk]


    t3=Counter(dist_lang)
    labels=[tk for tk in t3 if t3[tk]>50]
    fig1, ax1 = plt.subplots()
    ax1.pie([t3[tk]/3000000 for tk in labels], explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig("../artefacts/dist_lang.pdf", format="pdf")

