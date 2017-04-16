import numpy as np
def get_labels(names):
    positives = []
    negatives = []
    for x in range(len(names)):
        name = names[x]
        if "pos" in name:
            positives.append(1)
        else: #negative
            negatives.append(0)

    print ("Positives:",len(positives))
    print ("percent positive", float(len(positives))/float((len(positives)+ len(negatives))))
           
    print ("negatives: ", len(negatives))
    print ("percent negative", float(len(negatives))/float((len(positives)+ len(negatives))))
           
    return positives, negatives


feature_set = "larger"
names = np.load(feature_set + '_cell_filenames.npy')
get_labels(names)

feature_set = "medium"
names = np.load(feature_set + '_cell_filenames.npy')
get_labels(names)
