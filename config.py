import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print "loading config: %s failed" % (config_file)
        self.origin_graph_file = conf.get("Graph_Data", "origin_graph_file")
        self.train_graph_file = conf.get("Graph_Data", "train_graph_file")
        if conf.has_option("Graph_Data","label_file"):
            self.label_file = conf.get("Graph_Data", "label_file")
        

        ## embedding data
        self.embedding_filename = conf.get("Graph_Data", "embedding_filename")

        ## hyperparameter
        self.struct = [int(i) for i in conf.get("Model_Setup", "struct").split(',')]
        self.alpha = conf.getfloat("Model_Setup", "alpha")
        self.gamma = conf.getfloat("Model_Setup", "gamma")
        self.reg = conf.getfloat("Model_Setup", "reg")
        self.beta = conf.getfloat("Model_Setup", "beta")
        
        ## para for training
        self.batch_size = conf.getint("Model_Setup", "batch_size")
        self.epochs_limit = conf.getint("Model_Setup", "epochs_limit")
        self.learning_rate = conf.getfloat("Model_Setup", "learning_rate")
        self.display = conf.getint("Model_Setup", "display")

        self.DBN_init = True 
        self.dbn_epochs = conf.getint("Model_Setup","dbn_epochs")
        self.dbn_batch_size = conf.getint("Model_Setup","dbn_batch_size")
        self.dbn_learning_rate = conf.getfloat("Model_Setup","dbn_learning_rate")
        

        self.sparse_dot = False
        self.ng_sample_ratio = conf.getfloat("Model_Setup","ng_sample_ratio")
