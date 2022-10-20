import os
import nltk
import numpy as np
import pandas as pd
import inltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
from inltk.inltk import tokenize
from inltk.inltk import setup
#setup('mr')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

class FaqEngine:
    def __init__(self,  faqslist):
        self.faqslist = faqslist
        self.classifier = None

    def query(self, usr):

            #print(usr)

            df = pd.read_csv('data/dataset.csv')

            tagged_set = 'marathi.pos'
            word_set = indian.sents(tagged_set)
            count = 0
            #print(word_set)
            for sen in word_set:
                count = count + 1
                sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
                #print (sen)
            #print (count)

            #print(indian.tagged_words(fileids='marathi.pos', ))

            df1 = df
            #print(df1.groupby('Class').count())

            question_column = df1['Question']
            label_column = df1['Class']
            count = 0
            label_count = 0
            tokens = []
            labels = []
            for sen in question_column:
                count = count + 1
                tokenized_sen = tokenize(sen,'mr')
                cleansed_sent = []
                for word in tokenized_sen:
                    if word.startswith('▁'):
                        cleansed_word = word.replace('▁', "")
                        cleansed_sent.append(cleansed_word)
                #print (cleansed_sent)
                #print (count)
                tokens.append(cleansed_sent)
            for label in label_column:
                label_count = label_count + 1
                labels.append(label)
            #print(len(tokens))
            #print(len(labels))

            marathi_stopwords = ['आहे', 'या', 'आणि', 'व', 'नाही', 'आहेत', 'यानी', 'हे', 'तर', 'ते', 'असे', 'होते', 'केली', 'हा', 'ही', 'पण', 'करणयात', 'काही', 'केले', 'एक', 'केला', 'अशी', 'मात्र', 'त्यानी', 'सुरू', 'करून', 'होती', 'असून', 'आले', 'त्यामुळे', 'झाली', 'होता', 'दोन', 'झाले', 'मुबी', 'होत', 'त्या', 'आता', 'असा', 'याच्या', 'त्याच्या', 'ता', 'आली', 'की', 'पम', 'तो', 'झाला', 'त्री', 'तरी', 'म्हणून', 'त्याना', 'अनेक', 'काम', 'माहिती', 'हजार', 'सागित्ले', 'दिली', 'आला', 'आज', 'ती', 'तसेच', 'एका', 'याची', 'येथील', 'सर्व', 'न', 'डॉ', 'तीन', 'येथे', 'पाटील', 'असलयाचे', 'त्याची', 'काय', 'आपल्या', 'म्हणजे', 'याना', 'म्हणाले', 'त्याचा', 'असलेल्या', 'मी', 'गेल्या', 'याचा', 'येत', 'म', 'लाख', 'कमी', 'जात', 'टा', 'होणार', 'किवा', 'का', 'अधिक', 'घेऊन', 'परयतन', 'कोटी', 'झालेल्या', 'निर्ण्य', 'येणार', 'व्यकत']
            #print(marathi_stopwords)

            filtered_sentence = []
            filter_count = 0
            for sent in tokens:
                filtered_sent = []
                for word in sent:
                    if word not in marathi_stopwords:
                        filtered_sent.append(word)
                    else:
                        filter_count = filter_count+1
                filtered_sentence.append(filtered_sent)
            #print(filtered_sentence)
            #print(filter_count)

            questions = tokens
            sentences = []
            for arr in questions:
                sent = " ".join(arr)
                sentences.append(sent)
            X = sentences
            #print(X)

            #suffixes
            suffixes = ['झी', 'शी', 'रू', 'ळी', 'स्तीत', 'तो', 'झ्या', 'ळासाठी', 'झे', 'कडून', 'ले', 'ते', 'झा', 'लू', 'टासाठी', 'से', 'ल्यास', 'ने', 'मध्ये', 'सचे', 'सांत', 'रायचे', 'रावे', 'गसाठी', 'झे', 'सू', 'लू', 'साच्या', 'जनेत', 'ल्यास', 'वा', 'वू', 'वा', 'डीची', 'धीच', 'डू', 'रलेल्या', 'नासाठी', 'ळू', 'न्सवरील', 'सासाठी', 'सावर', 'ईला', 'ण्यासाठी', 'तीचे', 'वर', 'ळांवर', 'रण्याचे', 'स्तीचे', 'सच्या', 'लेले', 'नीकडे', 'रताना', 'ईसाठी', 'ट्ससाठी', 'सची', 'सचे', 'ण्यास', 'हिल्या', 'टला', 'ल्यामुळे', 'टाचा', 'क्तीचे', 'कांशिवाय', 'तांसाठी', 'अरसाठी', 'गांसाठी', 'जावर', 'णत्या', 'चे', 'त्याही', 'द्धतीने', 'रवरून', 'ण्यासाठी', 'सच्या', 'ळावर', 'रातील', 'लयात', 'द्रातून', 'शातील', 'यांमधून', 'न', 'परू', 'द्वारे', 'साची', 'शासाठी', 'र्डच्या', 'जूंची', 'धारकाने']
            replace = ['झ', 'स', 'र', 'ळ', 'स्त', 'त', 'झ', 'ळ', 'झ', '', 'ल', 'त', 'झ', 'ल', 'ट', 'स', 'ल', 'न', '', 'स', 'स', 'र', 'र', 'ग', 'झ', 'स', 'ल', 'स', 'जना', 'ल', 'व', 'व', 'व', 'ड', 'धी', 'ड', 'र', 'न', 'ळ', 'न्स', 'स', 'सा', 'ई', 'ण', 'त', '', 'ळ', 'र', 'स्त', 'स', 'ल', 'नी', 'र', 'ई', 'ट्स', 'स', 'स', 'ण्', 'हिल', 'ट', 'ल', 'ट', 'क्ती', 'क', 'ता', 'अर', 'गा', 'ज', 'णत', '', 'त', 'द्धत', 'र', 'ण', 'स', 'ळ', 'र', 'लय', 'द्र', 'श', 'य', 'न', 'पर', '', 'स', 'स', 'र्ड', 'जू', 'धारक']
            #print(len(suffixes))
            #print(len(replace))

            a = df['Question']
            cleaned = []
            for s in a:
                t = s.split(' ')
                temp = ''
                for i in t:
                    j = 0
                    for suffix in suffixes:
                                if i.endswith(suffix):
                                    b = i.rstrip(suffix)
                                    ind = suffixes.index(suffix)
                                    b = b + replace[ind]
                                    temp = temp + ' ' + b
                                    j = 1
                                    break
                    if j == 0:
                        temp = temp + ' ' + i      
                cleaned.append(temp)    
            #print(cleaned)       

            temp = labels
            y = []
            for label in temp:
                if label == 'बुकिंग':
                    y.append(0)
                if label == 'पेमेंट':
                    y.append(1)
                if label == 'रद्द करणे आणि परतावा धोरण':
                    y.append(2)
                if label == 'सामान':
                    y.append(3)
                if label == 'निकष':
                    y.append(4)
                if label == 'सामान्य':
                    y.append(5)
            #print(y)
            #len(y)

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
            # print(len(X_train))
            # print(len(X_test))
            # print(X_train)
            # print(y_train)

            vec = CountVectorizer()
            vec.fit(X_train)
            #print(len(vec.get_feature_names()))
            #print(vec.vocabulary_)

            X_transformed = vec.transform(X_train)
            #print(X_transformed)
            X_transformed.toarray()
            #print(pd.DataFrame(X_transformed.toarray(), columns= [vec.get_feature_names()]))

            # for test data
            X_test_transformed = vec.transform(X_test)
            #print(X_test_transformed)
            #print(pd.DataFrame(X_test_transformed.toarray(), columns= [vec.get_feature_names()]))

            # s1 = ['एअर इंडिया एक्सप्रेस कोणत्या प्रकारचे विमान उड्डाण करते?']
            # vec1 = vec.transform(s1).toarray()
            #print(vec1)
            #print('Question:' ,s1)

            self.classifier = MultinomialNB()
            self.classifier.fit(X_transformed, y_train)   

            # predict class
            y_pred_class = self.classifier.predict(X_test_transformed)

            # predict probabilities
            y_pred_proba = self.classifier.predict_proba(X_test_transformed)

            #print(metrics.accuracy_score(y_test, y_pred_class))

            #print(str(list(nb.predict(vec1))[0]).replace('0', 'बुकिंग').replace('1', 'पेमेंट').replace('2', 'रद्द करणे आणि परतावा धोरण').replace('3', 'सामान').replace('4', 'निकष').replace('5', 'सामान्य'))

            # print("User typed : " + usr)
            try:
                s1 = [usr]
                vec1 = vec.transform(s1).toarray()
                #print(vec1)
                print(str(list(self.classifier.predict(vec1))[0]).replace('0', 'बुकिंग').replace('1', 'पेमेंट').replace('2', 'रद्द करणे आणि परतावा धोरण').replace('3', 'सामान').replace('4', 'निकष').replace('5', 'सामान्य'))
                class_ = str(list(self.classifier.predict(vec1))[0]).replace('0', 'बुकिंग').replace('1', 'पेमेंट').replace('2', 'रद्द करणे आणि परतावा धोरण').replace('3', 'सामान').replace('4', 'निकष').replace('5', 'सामान्य')

                cos_sims = []

                questionset = df[df['Class'] == class_]
                #print(questionset)

                for question in questionset['Question']:
                    temp = [question]
                    #print(temp)
                    vec2 = vec.transform(temp).toarray()
                    sims = cosine_similarity(vec2, vec1)
                    cos_sims.append(sims)

                #print(cos_sims)

                if len(cos_sims) > 0:
                    ind = cos_sims.index(max(cos_sims))
                    #print(df['Answer'][questionset.index[ind]])
                    return (df['Answer'][questionset.index[ind]])
            except Exception as e:
                #print(e)
                return "क्षमस्व. तुमच्या प्रश्नाचे अनुसरण करू शकलो नाही [" + usr + "], कृपया दुसरा प्रश्न विचारा"

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.abspath( __file__ )),"data")
    faqslist = [os.path.join(base_path,"dataset.csv")]
    faqmodel = FaqEngine(faqslist)
    response = faqmodel.query("Hi")
    print(response)

