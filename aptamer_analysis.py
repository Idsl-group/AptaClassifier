from sklearn.feature_extraction.text import CountVectorizer
import RNA
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from functools import reduce

# a and g are purines
# c and t are pyrimidines

top_kmers = 50
loopPattern = re.compile(r'\(\.*\)')

def copy_from_index_list(indices: list[int], items: list[str]) -> list[str]:
    new = []
    for i in indices:
        new.append(items[i])
    return new

def pack_and_sort_descending(labels: list[str], values: list[int]) -> list[list[str] | list[float] | list[int]]:
    important_values = []
    important_labels = []
    for i in range(len(values)):
        important_values.append(values[i])
        important_labels.append(labels[i])
    package = zip(important_labels, important_values)
    package = sorted(package, reverse=True, key=lambda elem: elem[1])
    k = []
    v = []
    for i in package:
        k.append(i[0])
        v.append(i[1])
    return [k, v]

def countLoops(structure: str) -> list[int]:
    # hairpins
    hairpins = len(loopPattern.findall(structure))

    # internal loops
    forwardMatches =  re.findall(r"(?=(\(\.+\())", structure)
    reverseMatches =  re.findall(r"(?=(\)\.+\)))", structure)

    internal_loops = 0

    for i in forwardMatches:
        for j in reverseMatches:
            if len(i) == len(j):
                internal_loops += 1
                break

    return [hairpins, internal_loops] # type: ignore

def countLoops2(sequence: str) -> list[int]:
    structure, _ = RNA.fold(sequence)
    # hairpins
    hairpins = len(loopPattern.findall( structure))

    # internal loops
    forwardMatches =  re.findall(r"(?=(\(\.+\())", structure)
    reverseMatches =  re.findall(r"(?=(\)\.+\)))", structure)

    internal_loops = []

    for i in forwardMatches:
        for j in reverseMatches:
            if len(i) == len(j):
                internal_loops.append(i)
                break

    return [hairpins, internal_loops] # type: ignore

def countHairpins(sequence: str) -> int:
    structure, _ = RNA.fold(sequence)
    return len(loopPattern.findall(structure))

def countLoopSize(sequence: str) -> list[int]:
    structure, _ = RNA.fold(sequence)
    loops = loopPattern.findall(structure)
    if len(loops) == 0:
        return [-1]
    else:
        return list(map(lambda x: len(x) - 2, loops))
    
def getStemSize(sequence: str) -> list[int]:
    structure, _ = RNA.fold(sequence)
    hairpins = loopPattern.finditer(structure)
    loops = loopPattern.findall(structure)
    result = []
    for hairpin, loop in zip(hairpins, loops):
        stemLength = 1 # start at 1 since end() returns a )
        searchStart = hairpin.end()
        for i in range(searchStart, len(structure)):
            if (structure[i] == ')'):
                stemLength += 1
            if ((structure[i] == '.') and (i != len(structure) - 1)):
                break
        result.append(len(loop) - 2)
        result.append(stemLength)
    return result

def computeMFE(sequence: str) -> float:
    _, mfe = RNA.fold(sequence)
    return mfe

def computeMT(sequence: str) -> float:
    sequenceLength = len(sequence)
    adenosine = 0
    cytosine = 0
    guanosine = 0
    thymine = 0
    for s in sequence:
        match s:
            case "A":
                adenosine += 1
            case "C":
                cytosine += 1
            case "G":
                guanosine += 1
            case "T":
                thymine += 1
    if sequenceLength < 14:
        return (adenosine + thymine) * 2 + (guanosine + cytosine) * 4
    else:
        return 64.9 + 41 * (guanosine + cytosine - 16.4) / (adenosine + thymine + guanosine + cytosine)

def findPurines(sequence: str) -> float:
    numPurines = sequence.count("A") + sequence.count("G")
    return numPurines / len(sequence)

def findPyrimidines(sequence: str) -> float:
    numPyrimidines = sequence.count("C") + sequence.count("T")
    return numPyrimidines / len(sequence)

def computeGC(sequence: str) -> float:
    numGC = sequence.count("G") + sequence.count("C")
    return numGC / len(sequence)

def computeProperties(sequences: list[str]) -> pd.DataFrame:
    sequence_properties = []
    for i in sequences:
        ss, mfe = RNA.fold(i)
        sequence_properties.append([computeMT(i), mfe] + countLoops(ss))
    column_labels = ["mp", "mfe", "hairpins", "internal_loops"]

    return pd.DataFrame(sequence_properties, columns=column_labels)

# train the model here
with open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas:
        aptamerData = list(filter(lambda x: len(x) > 0, aptamers.read().split("\n")))
        dnaData = list(filter(lambda x: len(x) > 0, dnas.read().split("\n")))

        dnaSequences = dnaData + aptamerData

        X = computeProperties(dnaSequences)

        y = [0] * len(dnaData) + [1] * (len(aptamerData))

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=9) # 4853 negative samples; therefore shift positive indices up by 4853 (?)

        train_indices = []
        test_indices = []

        for i in range(len(y_train)):
            if y_train[i] == 1:
                train_indices.append(x_train.index.values[i] - 4853)
        
        for i in range(len(y_test)):
            if y_test[i] == 1:
                test_indices.append(x_test.index.values[i] - 4853)
        
        # get indices for train and test datas

        best_weight = round(len(dnaData) / (len(aptamerData)))

        xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight)

        xgb2.fit(x_train, y_train)

        # list(map(lambda x: x[1], xgb2.predict_proba(aptamer_X)))

train_set = copy_from_index_list(train_indices, dnaData)
test_set = copy_from_index_list(test_indices, dnaData)

def getTrainTestLabel(sequence: str) -> str:
    if sequence in train_set:
        return "TR"
    else:
        return "TE"

def getPositiveConfidence(sequence: str) -> float:
    properties = computeProperties([sequence])
    return xgb2.predict_proba(properties)[0][1]
    
def getTop50Kmers(k: int, sequences: list[str]) -> list[str]:
    cv = CountVectorizer(ngram_range=(k, k), lowercase=False, analyzer='char')
    kmers = cv.fit_transform(sequences).toarray() # type: ignore
    kmers = reduce(lambda x, y: x + y, kmers)
    result = pack_and_sort_descending(cv.get_feature_names_out(), kmers) # type: ignore
    return list(zip(result[0], result[1]))[0:50] # type: ignore

def getDominantKmer(k: int, sequence: str) -> list | int: # this needs a separate operation
    if len(sequence) <= k:
        return ['none', 'none']
    cv = CountVectorizer(ngram_range=(k, k), lowercase=False, analyzer='char')
    kmers = cv.fit_transform([sequence]).toarray()[0] # type: ignore
    names = cv.get_feature_names_out()
    dominance_list = []
    max_dominance = max(kmers)
    for i in range(len(kmers)):
        if kmers[i] == max_dominance:
            dominance_list.append(names[i])
    return ["#".join(dominance_list), max_dominance]

def get6MerInfo(sequence: str) -> list | int:
    return getDominantKmer(6, sequence)

def get5MerInfo(sequence: str) -> list | int:
    return getDominantKmer(5, sequence)

def get4MerInfo(sequence: str) -> list | int:
    return getDominantKmer(4, sequence)
    

operations = [getTrainTestLabel, getPositiveConfidence, len, computeMT, computeGC, findPurines, findPyrimidines, computeMFE, get4MerInfo, get5MerInfo, get6MerInfo, countHairpins, getStemSize] # list of all the operations we need
# list of 50 dominant kmers will be a separate operation by necessity
kmer_datasheet = list(map(lambda x: [getTop50Kmers(4, x), getTop50Kmers(5, x), getTop50Kmers(6, x)], [train_set]))
for sheet in kmer_datasheet:
    print(sheet, "#####")
# datasheet = list(map(lambda x: list(oper(x) for oper in operations), dnaData))

# for line in datasheet:
#     print(line)