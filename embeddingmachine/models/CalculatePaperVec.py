#coding=utf-8

import MySQLdb as mysql 
import math
import random
import re

def myLog2(x):
  return math.log(x) / math.log(2)

def cleanPaperInfo(paperDesc):
  paperDesc = paperDesc.lower()
  no_punctuation = re.sub(r'\([^\)]*\)|\[.*\]|\{.*\}|[^A-Za-z]', ' ', paperDesc)
  paperDescWordsList = no_punctuation.strip().split()
  return ' '.join(paperDescWordsList)

def readAllPaperInfo():
  conn = mysql.connect(host='172.22.0.26',user='recom',passwd='recom123',db='recommender',port=3306)
  cursor = conn.cursor()
  cursor.execute('select * from t_Arxiv1_paper')

  paperDescStrDict = dict()
  for row in cursor.fetchall():
    paperId = row[0].strip()
    title = row[1]
    abstract = row[2]

    #paperDesc = title if abstract is None else title + " " + abstract
    paperDesc = title 
    paperDescStrDict[paperId] = cleanPaperInfo(paperDesc)

  cursor.close()
  conn.close()    

  print 'Read paper desc done, paper num:', len(paperDescStrDict)
  return paperDescStrDict 

def normalize(vec):
  sum = 0.0 
  for e in vec:
    sum += e*e
  mode = math.sqrt(sum) 
  if mode > 1e-10:
    for i in range(len(vec)):
      vec[i] = vec[i] / mode
  return vec

def getWordIdf(paperDescStrDict):
  wordIdfCount = dict()
  wordIdf = dict()
  for paperId, paperDesc in paperDescStrDict.iteritems():
    paperWordsList = paperDesc.split()
    for word in set(paperWordsList):
      wordIdf.setdefault(word, 0)
      wordIdf[word] += 1 
  
  bigCount = 0
  with open('data/paper_title_nostem_big') as f:
    for paperTitle in f:
      paperWordsList = paperTitle.split()
      bigCount += 1
      for word in set(paperWordsList):
        wordIdf.setdefault(word, 0)
        wordIdf[word] += 1 

  for word in wordIdf:
    wordIdfCount[word] = wordIdf[word]
    wordIdf[word] = myLog2((len(paperDescStrDict) + bigCount) * 1.0 / wordIdf[word])

  with open('data/arxiv_word_idf_count', 'w') as f:
    f.write(str(len(paperDescStrDict) + bigCount) + ' 0' + '\n')
    for word, idf_count in wordIdfCount.iteritems():
      f.write(word + ' ' + str(idf_count) + '\n')

  with open('data/arxiv_word_idf', 'w') as f:
    for word, idf in wordIdf.iteritems():
      f.write(word + ' ' + str(idf) + '\n')
  print 'Calculate idf done'
  return wordIdf 

def readWordsVec():
  word2VecDict = dict()
  with open('data/arxiv_word_rep') as f:
    countInfo = f.readline()
    print countInfo
    allWordNum = int(countInfo.split()[0])
    dimNum = int(countInfo.split()[1])

    for line in f:
      fields = line.split()
      word = fields[0].strip()
      vec = list()
      vecDim = len(fields) - 1
      if vecDim != dimNum:
        raise Exception('Error, dim not match')
      for i in range(vecDim):
        vec.append(float(fields[i+1].strip()))
      vec = normalize(vec)
      word2VecDict[word] = vec

  if allWordNum != len(word2VecDict):
    raise Exception('Error, word num not match')

  print 'Read word vector done'
  return word2VecDict, dimNum
      
english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

def calculatePaperVec(word2VecDict, paperDescStrDict, wordIdf, vecDim):
  paperVecDict = dict()
  for paperId, paperDesc in paperDescStrDict.iteritems():
    paperWordsList = paperDesc.split()
    paperVec = [0.0] * vecDim
    wordCount = 0;
    for word in paperWordsList:
      """
      if (word not in word2VecDict) and (word not in english_stopwords) and len(word) >= 3:
        new_vec = [random.uniform(-1.0, 1.0) for i in range(vecDim)]
        normalize(new_vec)
        word2VecDict[word] = new_vec
      """
                                            
      if word in word2VecDict:
        wordWeight = wordIdf[word] if word in wordIdf else 1.0
        paperVec = [x+y*wordWeight for x,y in zip(paperVec, word2VecDict[word])] 
        wordCount += 1
    paperVecDict[paperId] = paperVec if wordCount == 0 else normalize(paperVec)

  print 'Calculate paper vector done'
  return paperVecDict

def calculatePaperVecAverage(word2VecDict, paperDescStrDict, vecDim):
  paperVecDict = dict()
  for paperId, paperDesc in paperDescStrDict.iteritems():
    paperWordsList = paperDesc.split()
    paperVec = [0.0] * vecDim
    wordCount = 0;
    for word in paperWordsList:
      if word in word2VecDict:
        paperVec = [x+y for x,y in zip(paperVec, word2VecDict[word])] 
        wordCount += 1
    paperVecDict[paperId] = paperVec if wordCount == 0 else normalize(paperVec)

  print 'Calculate paper vector sum done'
  return paperVecDict

def writeAllPaperVec(paperVecDict, vecDim):
  with open('data/arxiv_paper_rep', 'w') as f:
    f.write(str(len(paperVecDict)) + ' ' + str(vecDim) + '\n')
    for paperId, paperVec in paperVecDict.iteritems():
      f.write(paperId + ' ' + ' '.join(map(str, paperVec)) + '\n')

def writeWord2Vec(word2VecDict, vecDim):
  with open('data/arxiv_word_rep', 'w') as f:
    f.write(str(len(word2VecDict)) + ' ' + str(vecDim) + '\n')
    for word, wordVec in word2VecDict.iteritems():
      f.write(word + ' ' + ' '.join(map(str, wordVec)) + '\n')

if __name__ == "__main__":
  word2VecDict, vecDim = readWordsVec()
  print 'Vec dim:', vecDim
  paperDescDict = readAllPaperInfo()
  wordIdf = getWordIdf(paperDescDict)
  paperVecDict = calculatePaperVec(word2VecDict, paperDescDict, wordIdf, vecDim)
  print 'Paper num:', len(paperVecDict)
  writeAllPaperVec(paperVecDict, vecDim)
  #writeWord2Vec(word2VecDict, vecDim)
  
