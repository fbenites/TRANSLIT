import shelve

wiki_names = shelve.open("../artefacts/wiki_names.shf")

wkeys = list(wiki_names.keys())

wiki_names_cleaned = shelve.open("../artefacts/wiki_names_cleaned.shf")

for wkey in wkeys:
    items=[]
    for item in wiki_names[wkey]:

        try:
            if hasattr(item[1]._params[0].value._nodes[0],"_contents"):
                #        del wiki_names_cleaned[wkey]
                continue
        except:
            continue

        if wkey=="Kolkata–16":
            print(wkey)
            break

        try:
            item_t = str(item[1]._params[0].value._nodes[0]).split("|")[1][:-2]
        except:
            try:
                item_t = str(item[1]._params[0].value.nodes[0])
            except:
                continue
        items.append([item[0].split("-")[1],item_t])
    if len(items)>0:
        wiki_names_cleaned[wkey]=items

wiki_names.close()
wiki_names_cleaned.close()
        
