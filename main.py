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

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c",dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        options.config_file = 'config.ini'

    config = Config(options.config_file)
    
    origin_graph_data = Graph(config.origin_graph_file, config.ng_sample_ratio)
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio)
    
    #load label for classification
    #graph_data.load_label_data(config.label_file)
    
    config.struct[0] = train_graph_data.N
    
    model = SDNE(config)    
    model.do_variables_init(train_graph_data, config.DBN_init)

    epochs = 0
    batch_n = 0
    
    #graph_data = graph_data.subgraph(config.sample_method, config.sample_ratio)
    tt = time.ctime()
    fout = open(config.embedding_filename + '-' + tt +  "-log.txt","w") 
    while (True):
        mini_batch = train_graph_data.sample(config.batch_size)
        loss = model.fit(mini_batch)
        batch_n += 1
        #print "Epoch : %d, batch : %d, loss: %.3f" % (epochs, batch_n, loss)
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
                #check_link_reconstruction(embedding, train_graph_data, np.arange(100,1000,100))
                result = check_link_prediction(embedding, train_graph_data, origin_graph_data, [10, 100, 500, 1000, 10000])
                #data = origin_data.sample(origin_data.N, with_label = True)
                #check_multi_label_classification(model.get_embedding(data), data.label)
                print >> fout, epochs, result
            if epochs > config.epochs_limit:
                print "exceed epochs limit terminating"
                break
    embedding = model.get_embedding(train_graph_data.sample(origin_graph_data.N, do_shuffle = False))
    sio.savemat(config.embedding_filename + '-' + tt + '_embedding.mat',{'embedding':embedding})
    fout.close()
