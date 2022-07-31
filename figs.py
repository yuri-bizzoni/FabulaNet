# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt

from importlib import reload
import saffine
reload(saffine)
import numpy as np
import re
import saffine.multi_detrending as md
import saffine.detrending_method as dm
from scipy.stats import norm

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid =  SentimentIntensityAnalyzer()


def integrate(x):
    return np.mat(np.cumsum(x) - np.mean(x))

def normalize(ts, scl01 = False):
    ts01 = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
    ts11 = 2 * ts01 -1
    if scl01:
        return ts01
    else:
        return ts11
    
def optimal_bin(y):
    """ optimal number of bins for histogram
    src: https://academic.oup.com/biomet/article-abstract/66/3/605/232642
    """
    R = max(y)-min(y)
    n = len(y)
    sigma = np.std(y)
    return int(round((R * (n**(1./3.))) / (3.49 * sigma)))

# functions to produce figures
def figures(story_arc,sentimethod, workname, workid):
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    p = np.poly1d(np.polyfit(x, y, order))
    xp = np.linspace(0, len(x), len(x))

    #fig, ax = plt.subplots(2,1)

    plt.figure(figsize=(12,20))
    plt.subplot(311)
    X = np.mat([float(x) for x in story_arc])
    plt.plot(X.T,'-k', label = 'story arc')
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
            plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))
        except:
            print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            plt.plot(X.T,'-k', label = 'story arc')
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass
            #old_stdout = sys.stdout
            #if "/" in workid:
            #workid = workid.split("/")[1]
            #logname= str("./scores/error_"+workid+".log")
            #log_file = open(logname,"w")
            #sys.stdout = log_file
            #log_file.close()
            #sys.stdout = old_stdout
    # format
    #for i in range(2,5):
    #    _, trend_ww_1 = dm.detrending_method(X, w, i)
    #    plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))

    plt.title("$%s~Story~Arc~{}$".format(str(sentimethod)) % (workname))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$F(t)$')

    plt.subplot(312)
     # parameters
        
    for i in range(2,5):
        _, trend_ww_1 = dm.detrending_method(X, w, i)
        plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))

    #plt.title("$Passing~Story~Arc_{Syuzhet}$")
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$F(t)_{-1:1}$')
    
    plt.subplot(325)
    M = np.mean(story_arc)
    SD =  np.std(story_arc)
    n, bins, _ = plt.hist(story_arc, optimal_bin(story_arc) ,density = True , facecolor = 'gray', edgecolor = 'w')
    
    Y = norm.pdf(bins, M, SD)
    plt.plot(bins, Y, 'k-', linewidth=1.5)
    plt.ylabel('$Sentiment~score$')
    plt.ylabel('$Density$')

    
    plt.subplot(326)
    plt.plot(xp, p(xp), 'k-', linewidth = 2,zorder = 0)
    plt.scatter(x, y, c = 'r', s = 50, zorder = 1)
    plt.title('$H = {}$'.format(round(np.polyfit(x, y, 1)[0],2)))
    plt.xlabel('$Log(w)$')
    plt.ylabel('$LogF(w)$')
    
    plt.tight_layout()
    workname = re.sub("\W+"," ", workname.lower())
    workname = re.sub(" +", "_", workname)
    workname = workname +"_{}".format(sentimethod)
    workid = workid +"_{}".format(sentimethod)
    ####plt.savefig(os.path.join("fig", workid))
    #plt.savefig(workname+".png")
    
    plt.close()
    
    H = round(np.polyfit(x, y, 1)[0],2)
    return H


def get_Hurst(story_arc):
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    #fig, ax = plt.subplots(2,1)

    X = np.mat([float(x) for x in story_arc])
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
        except:
            print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass

        
    for i in range(2,5):
        _, trend_ww_1 = dm.detrending_method(X, w, i)
    
    H = round(np.polyfit(x, y, 1)[0],2)
    return H



def sentimarc_sid(text, untokd=True):
    if untokd:
        sents = nltk.sent_tokenize(text)
        #print(len(sents))
    else: sents = text
    arc=[]
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)['compound']
        arc.append(compound_pol)
    return arc


def sentimarc_lexicon(words, lexicon):
    senti_words = [w for w in words if w in lexicon.keys()]
    word_arc = [lexicon[w] for w in senti_words]
    return word_arc

def sentimarc_lexicon_oov(words, lexicon, neutral_value=.5):
    word_arc_u=[]
    for w in words:
        if w in lexicon.keys(): word_arc_u.append(lexicon[w])
        else: word_arc_u.append(neutral_value)
    return word_arc_u




def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_hierarchy_1D(X, labels=[], title='Hierarchical Clustering Dendrogram'):
    #df.index = df['newspaper_event']
    #X = np.array(df['distance']).reshape(-1, 1)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    model = model.fit(X)
    plt.figure(figsize=(20, 10))
    plt.title(title)
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=7, labels=labels) #df.index)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    
    

def plot_hierarchy_2D(X, labels=[], title='Hierarchical Clustering Dendrogram', node_col='newspaper_event', figsize=(20,10)):
    #df.index = df[node_col]
    #X = np.array(df[['X', 'Y']])

    # setting distance_threshold=0 ensures we compute the full tree.
    
    new_shape = []
    for el in X:
        new_shape.append([float(e) for e in el])

    new_shape = np.array(new_shape)
    
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(new_shape)
    #print(model.n_connected_components_, model.n_clusters_, model.labels_)
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=7, labels=labels)# df.index)
    plt.xticks(rotation=90)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    
    return fig 