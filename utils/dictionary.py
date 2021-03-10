def getBiWds(sentence):
    wds = []
    for id,wd in enumerate(sentence[:-1]):
        wds.append((str(wd)+'|'+str(sentence[id+1])))
    wds.append(str(sentence[-1])+'|end')
    return wds
def buildBiGramDictionary(X):
    wd2id = {}
    id2wd = {}
    id = 1
    for x in X:
        left = x[0]
        right = x[1]
        all_wds = getBiWds(left)+getBiWds(right)
        for wd in all_wds:
            if wd not in wd2id:
                wd2id[wd] = id
                id2wd[id] = wd
                id+=1
    return wd2id,id2wd,id