# SDNE
This repository provides a reference implementation of *SDNE* as described in the paper:<br>
> Structural Deep network Embedding.<br>
> Daixin Wang, Peng Cui, Wenwu Zhu<br>
> Knowledge Discovery and Data Mining, 2016.<br>
> <Insert paper link>

The *SDNE* algorithm learns a representations for nodes in a graph. Please check the [paper](http://www.kdd.org/kdd2016/subtopic/view/structural-deep-network-embedding) for more details. 

### Basic Usage
```
$ python main.py -c config/xx.ini
```
>noted: your can just checkout and modify config file or main.py to get what you want.
### Input
Your input graph data should be a **txt** file or a **mat** file and be under **GraphData folder** 
#### file format
The txt file should be **edgelist** and **the first line** should be **N** , the number of vertexes and **E**, the number of edges

The mat file should be the adjacent matrix. 

you can save your adjacent matrix using the code below

```
import scipy.io as sio
sio.savemat("xxx.mat", {"graph_sparse":your_adjacent_matrix})
```

It is recommended to use mat file and save the adjacent matrix in a sparse form.

#### txt file sample
	5242 14496
	0 1
	0 2
	4 9
	...
	4525 4526

> noted: The nodeID start from 0.<br>
> noted: The graph should be an undirected graph, so if (I  J) exist in the Input file, (J  I) should not.
### Citing
If you find *SDNE* useful in your research, we ask that you cite the following paper:

	@inproceedings{Wang:2016:SDN:2939672.2939753,
	 author = {Wang, Daixin and Cui, Peng and Zhu, Wenwu},
	 title = {Structural Deep Network Embedding},
	 booktitle = {Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 series = {KDD '16},
	 year = {2016},
	 isbn = {978-1-4503-4232-2},
	 location = {San Francisco, California, USA},
	 pages = {1225--1234},
	 numpages = {10},
	 url = {http://doi.acm.org/10.1145/2939672.2939753},
	 doi = {10.1145/2939672.2939753},
	 acmid = {2939753},
	 publisher = {ACM},
	 address = {New York, NY, USA},
	 keywords = {deep learning, network analysis, network embedding},
	} 



