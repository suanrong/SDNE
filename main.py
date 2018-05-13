'''
Reference implementation of SDNE

Author: Xuanrong Yao, Daixin wang

for more detail, refer to the paper:
SDNE : structral deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

#!/usr/bin/python2
# -*- coding: utf-8 -*-



from config import Config
from graph import Graph
from model.sdne import SDNE
from utils.utils import *
import scipy.io as sio
import time
import copy
from optparse import OptionParser
import os

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c",dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        raise IOError("no config file specified")

    config = Config(options.config_file)
    
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio)
   
    if config.origin_graph_file:
        origin_graph_data = Graph(config.origin_graph_file, config.ng_sample_ratio)

    if config.label_file:
        #load label for classification
        train_graph_data.load_label_data(config.label_file)
    
    config.struct[0] = train_graph_data.N
    
    model = SDNE(config)
    model.do_variables_init(train_graph_data)
    embedding = None
    while (True):
        mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
        if embedding is None:
            embedding = model.get_embedding(mini_batch)
        else:
            embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
        if train_graph_data.is_epoch_end:
            break
    

    epochs = 0
    batch_n = 0
    
    
    tt = time.ctime().replace(' ','-')
    path = "./result/" + config.embedding_filename + '-' + tt
    os.system("mkdir " + path)
    fout = open(path + "/log.txt","w")  
    model.save_model(path + '/epoch0.model')

    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    print "!!!!!!!!!!!!!"
    while (True):
        mini_batch = train_graph_data.sample(config.batch_size)
        loss = model.fit(mini_batch)
        batch_n += 1
        if train_graph_data.is_epoch_end:
            epochs += 1
            batch_n = 0
            loss = 0
            if epochs % config.display == 0:
                embedding = None
                while (True):
                    mini_batch = train_graph_data.sample(config.batch_size, do_shuffle = False)
                    loss += model.get_loss(mini_batch)
                    if embedding is None:
                        embedding = model.get_embedding(mini_batch)
                    else:
                        embedding = np.vstack((embedding, model.get_embedding(mini_batch))) 
                    if train_graph_data.is_epoch_end:
                        break

                print "Epoch : %d loss : %.3f" % (epochs, loss)
                print >>fout, "Epoch : %d loss : %.3f" % (epochs, loss)
                if config.check_reconstruction:
                    print >> fout, epochs, "reconstruction:", check_reconstruction(embedding, train_graph_data, config.check_reconstruction)
                if config.check_link_prediction:
                    print >> fout, epochs, "link_prediction:", check_link_prediction(embedding, train_graph_data, origin_graph_data, config.check_link_prediction)
                if config.check_classification:
                    data = train_graph_data.sample(train_graph_data.N, do_shuffle = False,  with_label = True)
                    print >> fout, epochs, "classification", check_multi_label_classification(embedding, data.label)
                fout.flush()
                model.save_model(path + '/epoch' + str(epochs) + ".model")
            if epochs == config.epochs_limit:
                print "exceed epochs limit terminating"
                break
    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    fout.close()
