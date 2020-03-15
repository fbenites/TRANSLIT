import numpy as np
import jellyfish
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#from joblib import Parallel, delayed
from multiprocessing import Process, Queue
import multiprocessing as mp
from multiprocessing import get_context
import sys
import unidecode


def create_char_matrix(data, uuids):
    cv=CountVectorizer(analyzer="char", ngram_range=(1,1))

    names_list=[]
    uids_list=[]
    for tk in uuids:
        for tj in data[tk]:
            names_lis.append(tj.split("_")[1])
            uids_list.append(tk)
    mat=cv.fit_transform(names_list)
    return mat, cv, uids_list


def generate_similar_pairs(args):
        print("started")
    #with get_context("spawn").Pool() as pool:
        nr_neg_diff, nr_uids,  uids, sel_uids, fpath = args
        data=pickle.load(open(fpath,"rb"))

        
        nr_neg_diff_c=0
        data_gen=[]
        c1=0
        uuids_used=set()
        print("loaded data", nr_neg_diff)
        sys.stdout.flush()
        while nr_neg_diff_c < nr_neg_diff and c1<len(sel_uids):
            c1 += 1
            if  nr_neg_diff_c% 50 == 0:
                print(nr_neg_diff_c, c1)
                ######

            sel_uid1 = sel_uids[c1]
            myind = uids[sel_uid1]
            mynextname = list(data[myind])
            mynextnames = [ (tk,unidecode.unidecode(tk.split("_")[1])) for tk in mynextname]
            #todo preselect based on characters occurrence

            find_next=0
            for nextsample in uids:
                if nextsample==myind or nextsample in uuids_used:
                    continue
                for i1,nextname in enumerate(list(data[nextsample])):
                    nextname_norm=unidecode.unidecode(nextname.split("_")[1])
                    for i1,mynextname_s in enumerate(mynextnames):

                        if  jellyfish.jaro_distance(mynextname_s[1], nextname_norm ) > 0.8:
                            x=mynextname_s[0]
                            y=nextname
                            
                            uuids_used.add(sel_uid1)
                            uuids_used.add(nextsample)
                            nr_neg_diff_c+=1
                            if  nr_neg_diff_c% 50 == 0:
                                print(jellyfish.jaro_distance(mynextname_s[1], nextname_norm ), mynextname_s[1], nextname.split("_")[1], mynextname_s[1], nextname_norm )
                            data_gen.append([x,y,0, myind, nextsample, mynextname_s[1], nextname_norm ])
                            find_next=1
                            break
                    if find_next==1:
                        break
                if find_next==1:
                        break

        print("returning",sel_uids[0])
        return data_gen

def generate_pairs(data, seed, nr_pos, nr_neg, nr_neg_diff, fpath="../artefacts/names_merged.pkl", nr_threads=16):
    np.random.seed(seed)
    pos_ids=set()
    neg_ids=set()
    neg_diff_ids=set()
    #create positive
    nr_pos_c=0
    nr_neg_diff_c=0
    nr_neg_c=0
    nr_uids=len(data)
    uids = list(data.keys()) 
    counter=0
    data_gen = []
    uuids_used = set()
    while nr_pos_c< nr_pos and counter<100:
        sel_uid = uids[np.random.choice(nr_uids)]
        if (sel_uid in pos_ids) or (len(data[sel_uid])<2):
            counter+=1
            continue

        uuids_used.add(sel_uid)
        pos_ids.add(sel_uid)
        
        x=np.random.choice(list(data[sel_uid]))
        y=np.random.choice(list(data[sel_uid]))
        diff=0
        c1=0
        while x==y:
            y=np.random.choice(list(data[sel_uid]))
            c1+=1
            if c1==100:
                break
        if c1==100:
            counter+=1
            continue
        
        counter=0        
        nr_pos_c+=1
        data_gen.append([x,y,1, sel_uid,sel_uid])

    if counter==100:
        print("could not find enough positives")
    counter=0

    while nr_neg_c< nr_neg and counter<100:
        #select uuid
        sel_uid1 = uids[np.random.choice(nr_uids)]
        if (sel_uid1 in neg_ids) or (len(list(data[sel_uid1]))<2):
            counter+=1
            continue


        sel_uid2 = uids[np.random.choice(nr_uids)]
        #select candidate from possible in each of the both uuids
        x=np.random.choice(list(data[sel_uid1]))
        y=np.random.choice(list(data[sel_uid2]))
        diff=0
        c1=0
        #check they are really different, else try again
        while x==y:
            y=np.random.choice(list(data[sel_uid2]))
            c1+=1
            if c1==100:
                break
        if c1==100:
            counter+=1
            continue                
        neg_ids.add(sel_uid1)
        neg_ids.add(sel_uid2)

        uuids_used.add(sel_uid1)
        uuids_used.add(sel_uid2)
        counter=0
        nr_neg_c+=1
        data_gen.append([x,y,0,sel_uid1,sel_uid2] )

    if counter==100:
        print("could not find enough negatives")
    counter=0

    c1=0
    #nr_threads=16
    quota=int(len(data)/nr_threads)+1
    pool = mp.Pool(nr_threads)
    uids_s = np.random.permutation(range(nr_uids))
    print("similar negative pairs", nr_threads, quota)
    results = pool.map(generate_similar_pairs,[((nr_neg_diff/nr_threads)+1, nr_uids,   uids, uids_s[quota*i:quota*(i+1)], fpath) for i in range(nr_threads)],1)
    #results=Parallel(n_jobs=nr_threads, backend="multiprocessing")(delayed(generate_similar_pairs)((nr_neg_diff/nr_threads)+1, nr_uids, data, uuids_used, uids, uids[nr_uids*i:nr_uids*(i+1)],  neg_ids) for i in range(int(nr_uids/100)))
    [data_gen.extend(tk) for tk in results]

    return data_gen



if __name__ == "__main__":
    if "merged" not in globals():
        merged=pickle.load(open("../artefacts/names_merged.pkl","rb"))
    generated_pairs=generate_pairs(merged, 1337, 100000, 100000, 100000)
    pickle.dump(generated_pairs,open("../artefacts/generated_pairs.pkl","wb"))
    gen_pairs=generated_pairs
    gen_par_new = [gen_pairs[tk] for tk in range(len(gen_pairs)-8)]+[tj for tk in range(len(gen_pairs)-8,len(gen_pairs)) for tj in gen_pairs[tk]]
    gen_par_new_checked = []
    for tk in gen_par_new:
        if len(tk)>8:
            gen_par_new_checked.extend(tk)
        else:
            gen_par_new_checked.append(tk)
    pickle.dump(gen_par_new_checked,open("../artefacts/gen_pairs.pkl","wb"))
