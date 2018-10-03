# -*- coding: utf-8 -*-

from __future__ import division
import web
import codecs
import collections
import unicodedata
import random
from math import sqrt
from math import log
import sys
import json
import os
import sys, numpy, scipy
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
import math
import string

web.config.debug = False

render = web.template.render(os.path.join(os.path.dirname(__file__), 'templates/'))

urls = ('/','Start',
        '/step2','Step2',
        '/step2b','Step2b',
        '/step3','Step3',
        '/example','Example')

######################################################################################################
class Start:
    """This class shows the homepage (refers to 'start.html').
    """
    def GET(self):
        
        return render.start()

######################################################################################################
class Example:
    """This class prepares the Finnish example.
    """
    def GET(self):
        web.setcookie('filename','finnish.txt')
        web.setcookie('userid','finnish')
        
        web.seeother('/step2')
        

#####################################################################################################
class Step2:
    """This class copies the input file to the server and classifies the symbols with Sukhotin's
    algorithm (refers to 'settings.html'). The GET and POST methods basicall do the same thing, the
    GET method assumes that the file has already been upload (it is needed for going back in the steps
    procedure).
    """
    
    
    def vow_cons_distinction(self):
          
        ##### get vowel consonant distinction #####
        
        sukh = Sukhotin(self.corpus)
        vowels = sukh.vowels
        cons = sukh.consonants
        freq = sukh.charFreq
        
        nr_tokens = sum(freq.values())
        
        #symbols = sorted([s for s in freq.keys() if unicodedata.category(s)[0] in ["L","P"]])
        symbols = sorted([s for s in freq.keys()])
        
        
        web.setcookie('symbols',",".join(symbols))
        
        symbolsInfo = [(s,"v",freq[s],c,unicodedata.category(s)) if s in vowels else (s,"c",freq[s],c,unicodedata.category(s)) 
            for c,s in enumerate(symbols)]
            
        ##### prepare output #####
        def makeTable(sym):
            vow_yes = ""
            cons_yes = ""
            rare = ""
            bgcolor = ""
            
            if sym[2] < (nr_tokens/1000) or sym[4][0] == "P":
                rare = "checked"
                bgcolor = 'style="background-color: #CA9697;"'
            else:
                if sym[1] == "v":
                    vow_yes = "checked"
                    bgcolor = 'style="background-color: #8CBD9A;"'
                if sym[1] == "c":
                    cons_yes = "checked"
                    
            return (sym[0].encode('utf-8'),sym[2],
            "<input type='radio' name='sym{}' value='yes' {} \
            onchange=\"statecheck('sym{}')\" >".format(sym[3],vow_yes,sym[3]),
            "<input type='radio' name='sym{}' value='no' {} \
             onchange=\"statecheck('sym{}')\">".format(sym[3],cons_yes,sym[3]),
            "<input type='radio' name='sym{}' value='ignore' {} \
             onchange=\"statecheck('sym{}')\">".format(sym[3],rare,sym[3]),
             "sym" + str(sym[3]),bgcolor
            )
            
        outputString = list()
        for s in symbolsInfo:
            outputString.append(makeTable(s))
        
        return outputString
    
    def GET(self):

        input_file_name = web.cookies().get('userid') + ".txt"
        userid = web.cookies().get('userid')
        symbols = web.cookies().get('symbols')
        
        # import all lines
        with codecs.open(os.path.join(os.path.dirname(__file__),
            "files/",input_file_name),'r','utf-8') as input_file:
          self.corpus = [line.strip() for line in input_file.readlines()]
          
        return render.settings(self.vow_cons_distinction())

        
    def POST(self):
        x = web.input(myfile={},_unicode=True)
        web.setcookie('filename',x['myfile'].filename )
        userid = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
        web.setcookie('userid',userid)
        
        ##### store file on server#####
        filedir = 'files/' # change this to the directory you want to store the file in.
        if 'myfile' in x: # to check if the file-object is created
            filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with #linux ones.
            filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename #with extension)
            filename = userid + '.txt'
            fout = open(os.path.join(os.path.dirname(__file__),filedir,filename),'w') # creates the file where the uploaded file #should be stored
            
            contents = x.myfile.file.read()
            fout.write(contents) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
        
        ##### get vowel consonant distinction #####
        self.corpus = contents.decode('utf-8').split('\n')
        
        return render.settings(self.vow_cons_distinction())
        

######################################################################################################
class Step2b:
    """This class takes the input from Step 2 and preprocesses the visualization"""
    def GET(self):
        return "Error: no POST data"
    def POST(self):
        coo = web.cookies(symbols='no symbols')
        symbolsList = coo.symbols
        symList = symbolsList.split(",")
        x = web.input()
        out = list()
        vowels = list()
        consonants = list()
        for c,s in enumerate(symList):
            out.append(s.decode('utf-8')  + " ["+ x["sym" + str(c)] + "]")
            if x["sym" + str(c)] == 'yes':
                vowels.append(s.decode('utf-8'))
            elif x["sym" + str(c)] == 'no':
                consonants.append(s.decode('utf-8'))
        
        method = x.method


        # PARAMETERS
        input_file_name = web.cookies().get('userid') + ".txt"
        ignore_bound = 0 # check whether C1 and C2 are not adjacent to consonants
        ignore_identicals = 0 # ignore successions of identical consonants (C1==C2)
        average = 0 # take average of phi values [i.e., (phi(C1,C2) + phi(C2,C1))/2) ]
        sorting = "ward"
        
        if "bound" in x and x.bound == 'yes': ignore_bound = 1
        if "ident" in x and x.ident == 'yes': ignore_identicals = 1
        if "average" in x and x.average == 'yes': average = 1
        if "sorting" in x: sorting = x.sorting
        
        print(sorting)

        # import all lines
        with codecs.open(os.path.join(os.path.dirname(__file__),
            "files/",input_file_name),'r','utf-8') as input_file:
          word_forms = [line.strip() for line in input_file.readlines()]
  
        # generate vowel and consonant classes
        non_symbol = "__"
  
        # make co-occurrence counts
        cases = collections.defaultdict(list)
        pairs = list()
        total_pairs = 0
        
        # if VH, then swap vowels and consonants
        # in what follows, the first and third symbols are always labelled as C
        # the middle symbol as V (regardless of what method is selected)
        if method == "vh":
            vowels,consonants = consonants,vowels
        
        # 1st step: go through all word forms to collect the cooccurrence counts
        for word_form in word_forms:
          # word = original word form, cv_form = reduced form to consonants and vowels
          word = list(word_form)
          cv_form = ["C" if c in consonants else "V" for c in word]
          word.insert(0,"##"),cv_form.insert(0,"##")
          word.append("##"),cv_form.append("##")
  
          # go through the whole word
          for i in range(1,len(word)-2):
    
            bound = 1
            identical = 0
            # if CVC structure found
            if cv_form[i-1] == "C" and cv_form[i] == "V" and cv_form[i+1] == "C":
              pair = (word[i-1],word[i+1])
              left_pair = (word[i-1],non_symbol)
              right_pair = (non_symbol,word[i+1])
              triple = word[i-1:i+2]
      
              # check whether C1 and C2 are  adjacent to consonants
              if cv_form[i-2] != "C" and cv_form[i+2] != "C":
                bound = 0
        
              # check whether C1 and C2 are identical
              if word[i-1] == word[i+1]:
                identical = 1

              if ( not (ignore_bound == 1 and bound == 1)) and \
              (not (ignore_identicals == 1 and identical == 1)):
                cases[pair].append((triple,word))
                cases[left_pair].append((triple,word))
                cases[right_pair].append((triple,word))
                total_pairs += 1
                pairs.append(pair)

        # create co-occurrence matrix
        symbols = {c[x] for c in cases.keys() for x in (0,1)}
        symbols = sorted(list(symbols))
        
        #print(symbols)

        co_occ = dict()

        for symbol1 in symbols:  
          curr_row = dict()
          for symbol2 in symbols:
            try:
              co_count = len(cases[(symbol1,symbol2)])
            except KeyError:
              co_count = 0
            curr_row[symbol2] = co_count
          co_occ[symbol1] = curr_row


        # 2nd step: calculate association values

        pmi_dict = dict()
        phi_dict = dict()
        prob_dict = dict()

        symbols.remove(non_symbol)

        for symbol1 in symbols:
          curr_row = dict()
          curr_row_prob = dict()
          curr_row_pmi = dict()
          for symbol2 in symbols:
            a = co_occ[symbol1][symbol2]
            b = co_occ[symbol1][non_symbol] - a
            c = co_occ[non_symbol][symbol2] - a
            d = total_pairs - b - c - a
    
            # phi coefficient
            try:
              phi = (a*d-b*c) / (sqrt((a+c)*(b+d)*(a+b)*(c+d)))
            except ZeroDivisionError:
              phi = 0
      
            # prob
            try:
              prob = a / float(a + b)
            except ZeroDivisionError:
              prob = 0
              
            # pmi
            all_symbols = [pair[g] for pair in pairs for g in (0,1)]
            prob_xy = a / total_pairs
            prob_x = all_symbols.count(symbol1) / len(all_symbols)
            prob_y = all_symbols.count(symbol2) / len(all_symbols)
            try:
              pmi = log(prob_xy,2) - log(prob_x,2) - log(prob_y,2)
              #pmi = log((prob_xy/prob_x * prob_y),2)
            except ValueError:
              pmi = 0
      
            # normalize between 0 and 1
            #phi = (phi + 1) / 2
            curr_row[symbol2] = phi
            curr_row_prob[symbol2] = prob
            curr_row_pmi[symbol2] = pmi
          phi_dict[symbol1] = curr_row
          prob_dict[symbol1] = curr_row_prob
          pmi_dict[symbol1] = curr_row_pmi
  



        if average == 1: # take the average of both values to make it symmetric
          for symbol1 in symbols:
            for symbol2 in symbols:
              phi_avg = \
                (phi_dict[symbol1][symbol2] + phi_dict[symbol2][symbol1]) / 2
      
              # normalize between 0 and 1
              #phi_avg = (phi_avg + 1) / 2
              phi_dict[symbol1][symbol2] = phi_avg
              phi_dict[symbol2][symbol1] = phi_avg
      


        phi_list = [[phi_dict[s1][s2] for s2 in symbols] for s1 in symbols]
        prob_list = [[prob_dict[s1][s2] for s2 in symbols] for s1 in symbols]
        pmi_list = [[pmi_dict[s1][s2] for s2 in symbols] for s1 in symbols]
        dataMatrix = numpy.array(phi_list)
        probMatrix = numpy.array(prob_list)
        pmiMatrix = numpy.array(pmi_list)

        # adapted from http://blog.nextgenetics.net/?e=44
        colHeaders = symbols[::]

        # take the square root with the correct sign
        def sqrt_abs(v):
            sign = 1
            if v < 0:
                sign = -1
            return sign*math.sqrt(abs(v))
            
        dataMatrix = numpy.array([sqrt_abs(d) for d in dataMatrix.flat]).reshape(dataMatrix.shape)
        probMatrix = numpy.array([sqrt_abs(d) for d in probMatrix.flat]).reshape(probMatrix.shape)
        pmiMatrix = numpy.array([sqrt_abs(d) for d in pmiMatrix.flat]).reshape(pmiMatrix.shape)

        # get the correct sorting by means of Ward's clustering
        def get_order(matrix,colHeaders):

            #calculate distance matrix and convert to squareform
            distanceMatrix = dist.pdist(matrix)
            distanceSquareMatrix = dist.squareform(distanceMatrix)

            #calculate linkage matrix
            linkageMatrix = hier.linkage(distanceSquareMatrix,method=str(sorting))

            #get the order of the dendrogram leaves
            heatmapOrder = hier.leaves_list(linkageMatrix)


            colHeaders = numpy.array(colHeaders)
            print(heatmapOrder)
            print(type(heatmapOrder))
            print(colHeaders)
            print(type(colHeaders))
            orderedColHeaders = colHeaders[heatmapOrder]
            return orderedColHeaders.tolist(),matrix #orderedDataMatrix
        
        orderedColHeaders,orderedDataMatrix = get_order(dataMatrix,colHeaders)
        orderedProbHeaders,orderedProbMatrix = get_order(probMatrix,colHeaders)
        orderedPmiHeaders,orderedPmiMatrix = get_order(pmiMatrix,colHeaders)

        # output everything as a json file
        outdict = dict()
        symlistarr = list()
        
        for c,s in enumerate(orderedColHeaders):
            symlistarr.append({"name":s,"phiOrder":c,"probOrder":orderedProbHeaders.index(s),
            "pmiOrder":orderedPmiHeaders.index(s)})
            
        outdict['symbols'] = symlistarr
        
        associations = list()
        
        for i in range(0,len(orderedColHeaders)):
            for j in range(0,len(orderedColHeaders)):
                currDict = dict()
                currDict['first'] = i
                currDict['second'] = j
                currDict['phi'] = \
                 dataMatrix[symbols.index(orderedColHeaders[i])][symbols.index(orderedColHeaders[j])]
                currDict['prob'] = \
                 probMatrix[symbols.index(orderedColHeaders[i])][symbols.index(orderedColHeaders[j])]
                currDict['pmi'] = \
                 pmiMatrix[symbols.index(orderedColHeaders[i])][symbols.index(orderedColHeaders[j])]
                associations.append(currDict)
                
        outdict['associations'] = associations
        
        output_file = web.cookies().get('userid') + ".json"
                
        with codecs.open(os.path.join(os.path.dirname(__file__),'static',output_file),
            'w','utf-8') as dataout:
                dataout.write(json.dumps(outdict,ensure_ascii=False)) 
    
        # redirect to visualization step
        raise web.seeother('/step3')

######################################################################################################
class Step3():
    """This class reads the filename for the json file and calls 'vis.html' to display the 
    visualization.
    """
    def GET(self):
         output_file = web.cookies().get('userid') + ".json"
         return render.vis(output_file)


######################################################################################################
class Sukhotin():
    """This class takes a list of word forms as input and computes the classification of all 
    symbols into vowels and consonants according to Sukhotin's algorithm.
    """
    
    def __init__(self,corpus):
        self.vowels = collections.defaultdict(int)
        self.consonants = collections.defaultdict(int)
        self.bigrams = collections.defaultdict(int)
        self.unigrams = collections.defaultdict(int)
        self.corpus = corpus
        self.vowelsFound = 0
        self.charFreq = collections.defaultdict(int)
        self.classify()
    
    def classify(self):
        for word in self.corpus:
            for i in range(0,len(word)-1):
                self.charFreq[word[i]] += 1
                if i == len(word)-2: self.charFreq[word[i+1]] += 1
                if word[i] != word[i+1]:
                    self.bigrams[(word[i],word[i+1])] += 1
                    self.bigrams[(word[i+1],word[i])] += 1
                    self.unigrams[word[i]] += 1
                    self.unigrams[word[i+1]] += 1
        numOfPhonemes = len(self.unigrams)
        
        for t in range(0,numOfPhonemes):
            maxPhoneme = self.maxSum()
            if self.vowelsFound: break
            self.vowels[maxPhoneme] = 1
            del self.unigrams[maxPhoneme]
            for phoneme in self.unigrams.keys():
                try:
                    self.unigrams[phoneme] -= 2 * self.bigrams[(maxPhoneme,phoneme)]
                except KeyError:
                    pass
        
        for phoneme in self.unigrams.keys():
            self.consonants[phoneme] = 1
    def maxSum(self):
        maxValue = -1
        maxPhoneme = ''
        for phoneme in self.unigrams.keys():
            if self.unigrams[phoneme] > maxValue:
                maxValue = self.unigrams[phoneme]
                maxPhoneme = phoneme
        if maxValue <= 0: self.vowelsFound = 1
        return maxPhoneme


######################################################################################################
if __name__ == "__main__":
   app = web.application(urls, globals())
   app.run()
