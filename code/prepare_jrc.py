

def normalize(name_string):
    s = name_string.replace('+', ' ').replace('-', ' ')
    s = s.lower()
    return s


def get_persons():
    import csv
    import collections
    from tqdm import tqdm
    import unidecode
    path = "entities"
    # text = open(path).read().lower()
    text = ""
    # symlink to the entities file from JRC-Names
    csvfile = open('../data/JRC/entities')
    csvreader = csv.reader(csvfile, delimiter='\t')
    persons = collections.defaultdict(list)
    chars = set()
    maxlenname = 0
    for line in tqdm(csvreader, leave=True):
        if line[0]=="#":
            continue
        tid = line[0]
        if tid == 0:
            continue
        tye = line[1]
        # organization
        if tye == 'O':
            continue
        lang = line[2]
        name = line[3] #unidecode.unidecode(line[3])
        persons[tid] += [[lang, normalize(name)]]
        maxlenname = max(len(name), maxlenname)
        for tk in name:
            chars.add(tk)
    return persons, chars


if __name__ == "__main__":
    persons,chars = get_persons()
    import shelve
    data = shelve.open("../artefacts/jrc_names_new.shf")
    for tid in persons:
       
        data[tid]=persons[tid]

    data.close()
 
