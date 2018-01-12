class Config(object):
    def __init__(self):
        ## graph data
        #self.file_path = "../Data_SDNE/Flickr2_1.txt"
        self.origin_file_path = "GraphData/dataset_UCLA/2/graph_sparse.mat"
        self.train_file_path = "GraphData/dataset_UCLA/2/traingraph_sparse.mat"
        #self.label_file_path = "GraphData/blogCatalog3-groups.txt"
        
        ## embedding data
        self.embedding_filename = "embeddingResult/2"

        ## hyperparameter
        self.struct = [None, 500, 100]
        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 100
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 10
        
        ## para for training
        #self.rN = 0.9
        self.batch_size = 64
        self.epochs_limit = 20
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 500
        self.dbn_batch_size = 64
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0 # negative sample ratio
        
        #self.sample_ratio = 1
        #self.sample_method = "node"
