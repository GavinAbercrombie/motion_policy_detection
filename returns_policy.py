### Matches Hansard HoC debate motions and motion quasi-sentences 
### to codes from the Comparative Manifesto Project.

import os, csv
from collections import Counter, OrderedDict
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

def lemmatizer(text):
    """Input: string of 1 or more words,
    output: string of corresponding lemmas"""
    lemmad = ''
    wnl = WordNetLemmatizer()
    lemma_list = [wnl.lemmatize(t) for t in word_tokenize(text)]
    for lem in lemma_list:
        lemmad = lemmad + lem + ' '
    return lemmad
  
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# load all the annotated UK manifesto quasi-sentences (qs) into a dictionary of format {code: [qs0, qs1, ..., qsn]} 
manifestos = {}
for filename in os.listdir('resources/manifestos/'):
    manifesto = csv.reader(open('resources/manifestos/'+filename), delimiter=',')
    for row in manifesto:
        if row[1].isdigit():
            if row[1] not in manifestos:
                manifestos[row[1]] = [row[0]]
            else:
                manifestos[row[1]].append(row[0])
                
# load hand-annotated motions:
data = csv.reader(open('resources/hand_annotated_motions_sub.csv'))
# create ordered dict of motions:
motions = OrderedDict()
for row in data:
    feats_no = len(row) # no. of feature types for analysis (1 = title, 2 = motion, 3 = speech)
    idee = row[0]
    if idee not in motions: # create dict entry for fist example of each code
        motions[idee] = [[row[f] for f in range(1, feats_no)]] 
    else: # add subsequent examples to corresponding code values
        motions[idee].append([row[f] for f in range(1, feats_no)])
        
# get aggregate annotations and create list of sentences:
sentences = []
agg_anns = [] # for each motion, we're going to put the aggregation of all the qs annotations here
for k, v in motions.items():
    anns = [] # temporary list of annotations. Next, find most common code in list
    for i in v:
        if i[-1] != '000': # allow non-000 codes to dominate if 000 is majority code
            anns.append(i[-1])
        sentences.append(i)
    if len(anns) == 0:
        anns.append('000')
    agg_anns.append(Counter(anns).most_common(1)[0][0])
no_qs = len(sentences) # store number of qss for use in splitting tfidf matrix

# load manifesto project coding scheme:
manifesto_codes = csv.reader(open('resources/manifesto_codes.csv'))
codes = {}
for line in manifesto_codes:
    codes[line[0]] = line[1]
    
# create dict of coded manifestos
manifesto_codes_dict = {}
uk_codes = [] # find out which codes are actually used in uk manifestos
for csv_file in os.listdir('resources/manifestos'):
    manifests = csv.reader(open('resources/manifestos/' + csv_file))
    for row in manifests:
        if row[1].isdigit():
            if row[1] not in manifesto_codes_dict:
                manifesto_codes_dict[row[1]] = [codes[row[1]], row[0].replace("\u2022",""), row[1]] # remove bullet points
                uk_codes.append(row[1])
            else:
                manifesto_codes_dict[row[1]][1] += ' ' + row[0].replace("\u2022","")
                
# put together data from 'texts' and 'manifesto_codes_dict'
for k, v in manifesto_codes_dict.items():
    sentences.append(v)
# extract data for testing:
all_data = []
class_labels = []
for i in sentences:
    all_data.append(LemmaTokenizer(i[0]) + ' ' + LemmaTokenizer(i[1]))
    class_labels.append(i[-1])
print('No. of "true" class labels (for both motion sentences + manifesto excerpts):',len(class_labels))

# extract and combine strings from lists for testing:
all_data = [] # store test strings here
class_labels = []
for i in sentences:
    all_data.append(LemmaTokenizer(i[0]) + ' ' + LemmaTokenizer(i[1])) # needs to be adapted to handle more input
    class_labels.append(i[-1])
    
# extract data for testing:
all_data = []
class_labels = []
for i in sentences:
    lemmastring = lemmatizer(i[0])
    for j in range(1,2):
        lemmastring += ' ' + lemmatizer(i[j])
    all_data.append(lemmastring)
    class_labels.append(i[-1])
    
# get motion + manifesto code text and perform tf-idf:
tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)

# need to reorganise to get motion + qs level results
score = 0
scores_dict = OrderedDict()
count = 0
seen_sents = 0 
for k, v in motions.items():
    print('******************************')
    title = v[0][0]
    print(title, agg_anns[count])
    no_sents = len(v) # no. of sentences in the motion
    all_results = []
    for i in range(no_sents):
        sentence_index = seen_sents + i
        #print(sentence_index)
        cos_sim = cosine_similarity(tfidf_matrix[sentence_index], tfidf_matrix)[0]
        # get largest cos sim scores:
        top5 = heapq.nlargest(5, range(no_qs,len(cos_sim)), key=cos_sim.__getitem__)
        results = []
        for result in top5:
            results.append(sentences[result][2])
        all_results.append(results[0])
    agg_res = Counter(all_results).most_common(1)[0][0]
    print(agg_res)
    if agg_anns[count] == agg_res:
        score += 1
    seen_sents += no_sents
    count += 1
print('No. of matches:',score)

print('Accuracy:',(score/len(motions))*100,'%')
