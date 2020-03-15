import shelve
import data_generator


def print_table(headers, rows):
    print("\\begin{table}\n\\begin{tabular}{"+"".join(["|c|"]*len(headers))+"}\n")
    print("&".join(headers)+"\\\\\n")
    for row in rowHeaders:
        print(row+"\\\\\n")
    print("\\end{tabular}\n\\caption{}\n\\label{}\n\\end{table}\n")
    

if "amazon_names" not in globals():
    amazon_names_s = shelve.open("../artefacts/amazon_data.shf")
    amazon_names = dict([(tk,amazon_names_s[tk]) for tk in amazon_names_s])
    amazon_names_s.close()

    nal=list(amazon_names.keys())

#normalize
amazon_names_normed = {}
for i1,k in enumerate(amazon_names):
    amazon_names_normed[i1]=['en_'+k]+ [tk[0]+'_'+tk[1] for tk in amazon_names[k]]




#select 20'000 positive, negative and similar
import os
fpath_an="../artefacts/amazon_normed.pkl"
if not os.path.exists(fpath_an):
    import pickle
    pickle.dump(amazon_names_normed, open(fpath_an,"wb"))
ama_nor_pairs=data_generator.generate_pairs(amazon_names_normed, 42, 7000, 7000, 7000, fpath=fpath_an)


if "wiki_names_cleaned" not in globals():
    wiki_names_cleaned_s = shelve.open("../artefacts/wiki_names_cleaned.shf")
    wiki_names_cleaned = dict([(tk,wiki_names_cleaned_s[tk]) for tk in wiki_names_cleaned_s])
    wiki_names_cleaned_s.close()


#normalize
wiki_names_cleaned_normed = {}
for i1,k in enumerate(wiki_names_cleaned):
    wiki_names_cleaned_normed[i1]=['en_'+k]+ [tk[0]+'_'+tk[1] for tk in wiki_names_cleaned[k]]




#select 20'000 positive, negative and similar
import os
fpath_an="../artefacts/wiki_normed.pkl"
if not os.path.exists(fpath_an):
    import pickle
    pickle.dump(wiki_names_cleaned_normed, open(fpath_an,"wb"))
wiki_nor_pairs=data_generator.generate_pairs(wiki_names_cleaned_normed, 42, 7000, 7000, 7000, fpath=fpath_an)




if "jrc_names" not in globals():
    jrc_names_s = shelve.open("../artefacts/jrc_names_new.shf")
    jrc_names = dict([(tk,jrc_names_s[tk]) for tk in jrc_names_s])
    jrc_names_s.close()


jrc_names_normed = {}
for i1,k in enumerate(jrc_names):
    jrc_names_normed[i1]= [tk[0]+'_'+tk[1] for tk in jrc_names[k]]




#select 20'000 positive, negative and similar
import os
fpath_an="../artefacts/jrc_names_normed.pkl"
if not os.path.exists(fpath_an):
    import pickle
    pickle.dump(jrc_names_normed, open(fpath_an,"wb"))
jrc_nor_pairs=data_generator.generate_pairs(jrc_names_normed, 42, 7000, 7000, 7000, fpath=fpath_an)



# together corpus
if "gen_pars" not in globals():
    import pickle
    translit_gen_pars = pickle.load(open("../artefacts/gen_pairs.pkl","rb"))

# get only 21k from the translit_gen_pars


import numpy as np

np.random.seed(42)

rids = np.random.permutation(range(len(translit_gen_pars)))[:21000]
translit_pars = [translit_gen_pars[tk] for tk in rids]

# CV on each dataset
from sklearn.model_selection import KFold


datasets_names = ["jrc", "wiki","ama","translit"]

datasets = [ jrc_nor_pairs, wiki_nor_pairs,
             ama_nor_pairs, translit_pars]
scores=[]
for i1,ds in enumerate(datasets):
    kf = KFold(n_splits=10)
    np.random.seed(42)
    score_ds=[]
    for train_index, test_index in kf.split(np.random.permutation(range(len(ds)))):
        score_ds.append(do_classification(train_index, test_index, [ds[tk][2] for tk in train_index],
                          [ds[tk][2] for tk in test_index],ds))
    scores.append(score_ds)


rows=[]
for tk,tj,tn in zip(np.mean(scores,1),np.var(scores,1),datasets_names ):
    row=tn
    for tl,tm in zip(tk,tj):
        row+="&"+"{:0.3f}".format(tl)+"$\\pm$"+"{:0.3f}".format(tm)
    rows.append(row)
    
print_table(["Dataset","RF","SVM","SVM Char"],rows)
# train on one test on other


#
scores_base=[]

for i1,ds in enumerate(datasets[:-1]):
    scores_base.append(classification_experiments.do_classification(range(len(translit_pars)), range(len(translit_pars),len(translit_pars)+len(ds)),
                                                                    [translit_pars[tk][2] for tk in range(len(translit_pars))],
                                                                    [ds[tk][2] for tk in range(len(ds))],translit_pars+ds))


scores_mixed=[]
for i1,ds in enumerate(datasets):
    kf = KFold(n_splits=10)
    np.random.seed(42)
    score_ds=[]
    for train_index, test_index in kf.split(np.random.permutation(range(len(ds)))):
        score_ds.append(classification_experiments.do_classification(list(range(len(translit_pars)))+(len(translit_pars)+train_index).tolist(), len(translit_pars)+test_index,
                                                                    [translit_pars[tk][2] for tk in range(len(translit_pars))]+[ds[tk][2] for tk in train_index],
                                                                       [ds[tk][2] for tk in test_index],translit_pars+ds))
    scores_mixed.append(score_ds)


rows=[]
for tk,tj,tn in zip(np.mean(scores_mixed,1),np.var(scores_mixed,1),datasets_names ):
    row=tn
    for tl,tm in zip(tk,tj):
        row+="&"+"{:0.3f}".format(tl)+"$\\pm$"+"{:0.3f}".format(tm)
    rows.append(row)
    
print_table(["Dataset","RF","SVM","SVM Char"],rows)

scores_increasing=[]

rids = np.random.permutation(range(len(translit_gen_pars)))
for i in [1000,10000,20000,40000,80000,160000]:
    kf = KFold(n_splits=10)
    np.random.seed(42)
    translit_pars_t = [translit_gen_pars[rids[tk]] for tk in range(i)]
    ds=translit_pars_t
    score_ds=[]
    for train_index, test_index in kf.split(np.random.permutation(range(len(ds)))):
        score_ds.append(classification_experiments.do_classification(train_index, test_index, [ds[tk][2] for tk in train_index],
                          [ds[tk][2] for tk in test_index],ds))
    scores_increasing.append(score_ds)
        

rows=[]                                                                                                                       
for tk,tj,tn in zip(np.mean(scores_increasing,1),np.var(scores_increasing,1),["1k","10k","20k","40k","80k","160k"] ):                                          
    row=tn                                                                                                                    
    for tl,tm in zip(tk,tj):                                                                                                  
        row+="&"+"{:0.3f}".format(tl)+"$\\pm$"+"{:0.3f}".format(tm)                                                           
    rows.append(row)                                                                                                          
                                                                                                                              
print_table(["Dataset","RF","SVM","SVM Char"],rows)

import matplotlib.pyplot as plt
import numpy as np

ind=[1,2,3,4,5,6]
p0=plt.errorbar(ind, np.mean(scores_increasing,1)[:,0], np.var(scores_increasing,1)[:,0],  fmt='-o')

p1=plt.errorbar(ind, np.mean(scores_increasing,1)[:,1], np.var(scores_increasing,1)[:,1],  fmt='-o')

p2=plt.errorbar(ind, np.mean(scores_increasing,1)[:,2], np.var(scores_increasing,1)[:,2],  fmt='-o')
plt.xticks(ind, ('1k', '10k', '20k', '40k', '80k','160k'))
plt.legend((p0[0],p1[0], p2[0]), ('RF', 'SVM', "SVM-TF-IDF"))
plt.title("Dependency of the Accuracy on Increasing Pair Sample")
plt.savefig("inc_plot.pdf")
