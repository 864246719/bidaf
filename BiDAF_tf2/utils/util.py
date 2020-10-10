
import json

import os
import re
import numpy as np
import jieba


def print_all_file_path(init_file_path, keyword):
    paths = []
    for cur_dir, sub_dir, included_file in os.walk(init_file_path):
        if included_file:
            for file in included_file:
                if re.search(keyword, file):
                    paths.append(cur_dir + "\\" + file)
    return paths

def trainByW2V():
    paths1 = [r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\homework_02_code\BiDAF_tf2\data\squad\train-v1.1.json']

    allLine =  []
    for path in paths1:
        with open(path,'r',encoding='utf-8') as file:
            for line in file:
                allLine.append(line)

    allText = []
    for i in range(len(json.loads(allLine[0]).get('data')[0].get('paragraphs'))):
        for j in range(len(json.loads(allLine[0]).get('data')[0].get('paragraphs')[i-1].get('qas'))):
            allText.append(json.loads(allLine[0]).get('data')[0].get('paragraphs')[i-1].get('qas')[j-1].get('question'))
            allText.append(json.loads(allLine[0]).get('data')[0].get('paragraphs')[i-1].get('context'))
            allText.append(json.loads(allLine[0]).get('data')[0].get('paragraphs')[i-1].get('qas')[j-1].get('answers'))
            


    ###加载第二个文件的内容
    # paths
    path =r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\homework_02_code\BiDAF_tf2\data\squad'
    paths = print_all_file_path(path, ".json")

    allLine =  []
    for path in paths[1:]:
        with open(path,'r',encoding='utf-8') as file:
            for line in file:
                allLine.append(line)

    for line in allLine:
        if line is not None:
            if json.loads(line).get('documents')[0].get('paragraphs') is not None:
                allText.append(json.loads(line).get('documents')[0].get('paragraphs'))
            if json.loads(line).get('documents')[0].get('segmented_title') is not None:    
                allText.append(json.loads(line).get('documents')[0].get('segmented_title'))
            if json.loads(line).get('documents')[0].get('segmented_paragraphs') is not None:
                allText.append(json.loads(line).get('documents')[0].get('segmented_paragraphs'))
            if json.loads(line).get('documents')[0].get('title') is not None:
                allText.append(json.loads(line).get('documents')[0].get('title'))





    # ######################训练词向量

    # fileRead = []
    # for file in newpaths:
    #     with open(file,'r',encoding='utf-8') as fileTrainRaw:
    #         for line in fileTrainRaw:
    #             fileRead.append(line)
    # print(fileRead)



    # print(allText[0])
    # print(allText[1])
    # print(allText[2])
    # print(allText[3])



    fileSegWordDonePath = r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\preprocessed\fileseg1.txt'
    with open(fileSegWordDonePath,'w',encoding='utf-8') as fW:
        for text in allText:
            fileTrainSeg = []
            text = str(text)
            if text is not None and text!='[]' :
                sentence=jieba.lcut(text)
                for word in sentence:
                    fileTrainSeg.append(word)
                fW.write(" ".join(fileTrainSeg))
                fW.write('\n')






    ###################################################################################################

    """
    gensim word2vec获取词向量
    """

    import warnings
    import logging
    import os.path
    import sys
    import multiprocessing

    import gensim
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    from gensim.models.word2vec import PathLineSentences

    # 忽略警告
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


    program = os.path.basename(sys.argv[0]) # 读取当前文件的文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    inp = r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\preprocessed\fileseg1.txt'
    out_model = r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\preprocessed\corpusSegDone_1.model'
    out_vector = r'C:\Users\Administrator.DESKTOP-BN41LK7\Desktop\preprocessed\corpusSegDone_1.vector' 
    model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(out_model)
    model.wv.save_word2vec_format(out_vector, binary=False)
    return model2.wv.vectors