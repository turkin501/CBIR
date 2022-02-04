
import os
from image_search_pipeline.descriptors import DetectAndDescribe
from image_search_pipeline.information_retrieval import BagOfVisualWords
from image_search_pipeline.information_retrieval import Searcher
from image_search_pipeline.information_retrieval import dist
from scipy.spatial import distance
from image_search_pipeline.information_retrieval import chi2_distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import numpy as np
import progressbar
import argparse
import pickle
import imutils
import json
import cv2
from image_search_pipeline import ResultsMontage
import argparse

arg = argparse.ArgumentParser()
arg.add_argument("-g", "--groundtruth", required = True, help = "Path to the directory of groundtruth file")
arg.add_argument("-i", "--idf", required = True, help = "Path to the directory of idf file")
arg.add_argument("-c", "--codebook", required = True, help = "Path to the directory of vocab file")
arg.add_argument("-b", "--bovw_db", required = True, help = "Path to the bag-of-visual-words database")
arg.add_argument("-f", "--features_db", required = True, help = "Path to the feature database")
arg.add_argument("-d", "--dataset", required = True, help = "Path to the directory of indexed images")
arg.add_argument("-r", "--result", required = True, help = "Path to the directory of the result of queries")

queries=[]
for (dirpath, dirnames, filenames) in os.walk(arg["groundtruth"]):
  for filename in filenames:
    query=filename[:filename.find('.')]
    if('_query' in query):
      queries.append(query)

def find_image(img, x, y, w, h,num):

    # initialize the keypoint detector, local invariant descriptor, descriptor pipeline,
    # distance metric, and inverted document frequency array
    detector = FeatureDetector_create("BRISK")
    descriptor = DescriptorExtractor_create("RootSIFT")
    dad = DetectAndDescribe(detector, descriptor)
    distanceMetric = chi2_distance
    idf = None

    # if the path to the inverted document frequency is array was supplied, then load
    # idf array and update the distance metric
    idf = pickle.loads(open(arg["idf"], "rb").read())
    distanceMetric = distance.cosine

    # load the codebook vocabulary and initialize the BOVW transformer
    vocab = pickle.loads(open(arg["codebook"], "rb").read())
    bovw = BagOfVisualWords(vocab)

    # load the query image and process it
    queryImage = cv2.imread(img)[y:w, x:h]
    queryImage = imutils.resize(queryImage, width = 320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct a bag-of-visual-word from  it
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    # connect to redis and perform the search
    redisDB = Redis(host = "localhost", port = 6379, db = 0)
    searcher = Searcher(redisDB, arg["bovw_db"],arg["features_db"], idf = idf, distanceMetric = distanceMetric)
    sr = searcher.search(hist, numResults = num)
    print("[INFO] search took: {:.2f}s".format(sr.search_time))

    # initialize the results montage
    montage = ResultsMontage((240, 320), 5, num)

    # loop over the individual results
    result_list = []
    for (i, (score, resultID, resultsIdx)) in enumerate(sr.results):
        result_list.append(resultID.decode("utf-8").rstrip('.jpg'))
    
    return result_list

def compute_ap(results, oks, junks, num):
  sum = 0
  ts = 0
  ms = 0
  count = 0
  for i in range(num):
      if results[i] not in junks:
          count +=1
          if results[i] not in oks:
              ms += 1
          else:
              ts += 1
              ms +=1
          sum += ts/ms
  return sum/count

ap_queries=[]

for query in queries:
  query_path=arg["groundtruth"]+'/'+query+'.txt'
  query_info = open(query_path, 'r')
  line_query = query_info.read()
  query_image, x, y, h, w = line_query.split()[0] + '.jpg', int(float(line_query.split()[1])), int(float(line_query.split()[2])), \
        int(float(line_query.split()[3])), int(float(line_query.split()[4]))
  oks=[]
  junks=[]
  with open(arg["groundtruth"]+'/'+query.rstrip('_query')+'_ok.txt', 'r') as f:
    tmp = f.readlines() 
    for s in tmp:
      oks.append(s.strip())
  with open(arg["groundtruth"]+'/'+query.rstrip('_query')+'_junk.txt', 'r') as f:
    t = f.readlines() 
    for s in t:
      junks.append(s.strip())
  result_list=find_image(arg["dataset"]+'/'+query_image[6:query_image.rfind('_')]+'/'+query_image,x,y,w,h,len(junks))
  with open(arg["result"]+'/'+query+'.txt', 'w') as f:
    f.write('\n'.join(result_list))
  ap=round(compute_ap(result_list,oks,junks,len(junks)),2)
  print('Average Precision: '+str(ap))
  ap_queries.append(ap)

with open('AP.txt','w') as f:
  f.write('\n'.join(ap_queries))