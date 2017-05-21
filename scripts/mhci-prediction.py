
# coding: utf-8

# # MHC Class I-Peptides Binding Affinity Prediction

# In this tutorial, we will build a convolutional network + LSTM to do MHC clss I binding affinity prediction from a Benchmark data set from [Kim _et al_. 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241)

# ## Prerequisite

# ### Define paths

# In[1]:

# The path corresponds to where you clone the bioitworld2017 repository
PRJ = "/home/bioit/bioitworld2017"


# ### Import modules

# In[2]:

from __future__ import print_function
from Bio.Alphabet import IUPAC
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as stats
import sys
from sklearn.metrics import roc_auc_score

get_ipython().magic(u'matplotlib inline')


# We will use the [Keras](http://keras.io) library to build the deep learning model. It is a very straightforward wrapper around the popular tensor-based library [Theano](http://deeplearning.net/software/theano/introduction.html) and Google's [TensorFlow](https://www.tensorflow.org/). The user just need to connect the layers, the library will build the low-level parameters and operations by calling the backend libraries. Eventually the model is translated into C++ code for speed.

# In[21]:

import keras
from keras.layers import (
    Activation, BatchNormalization, Conv1D, MaxPooling1D,
    Dense, Dropout, Embedding, LSTM, 
)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.regularizers import l1_l2


# ## Read data set
# 
# Most time we are spending when building a machine learning model is in two things:
# 
#   1. Cleaning and consolidating data set
#   2. Tuning parameters
#   
# I have previously re-formatting the MHC class I benchmark data set into training-ready format. Everything is wrapped within a [`pickle`](https://docs.python.org/2/library/pickle.html) file. 

# In[11]:

df, X, y, meta, sample_weights,AA_MAP = pickle.load(
    open(os.path.join(PRJ, "data/mhci-v0.2.pkl"), "rb")
)


# The pickel file contains following information:

# #### Raw data frame: `df`
# 
# The original data frame read from the data set.

# In[4]:

df.sample(10)


# #### Data matrices: `X` and `y`
# 
# The transformed `numpy.ndarray` that can directly fed to the model
# 
#   * `X` : the encoded amino acid sequence
#   * `y` : a vector of size (`num_samples`) with the log10 binding affinity for each peptide to the HLA subtype

# In[7]:

print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))


# #### Meta data: `meta`
# 
# A data frame that connects the matrices back to the annotation, including the randomly split training and test set label (20% of test).

# In[8]:

meta.sample(10)


# #### Amino acid code book: `AA_MAP`
# 
# The mapping between the code in `X` and the amino acid.

# In[9]:

IUPAC.extended_protein.letters


# In[12]:

AA_MAP


# #### Sample weights: `sample_weights`
# 
# From the `meta` data we can see there are some measurements with inequalities, _e.g._ > 20,000. If our objective is to get the minimum error from the predicted binding affinities, we have to down-weight these samples. For simplicity, we down-weighted these samples' weights by half, as you also see in `meta`.

# In[13]:

sample_weights


# ### Subsetting and splitting the data set
# 
# Let's first check how many samples we have for each HLA subtype:

# In[14]:

meta.groupby("mhc").size()


# For a lot of HLA types, we don't have a lot of samples. The current state-of-art approaches such as [NetMHC](http://www.cbs.dtu.dk/services/NetMHC/) train one model for one HLA subtype. Here let's focus on the largest one: HLA-A0201. We also only use peptides that's shorter than 15 amino acids.
# 
# We also want to split the data set into training and testing based on the index in the metadata. The testing set will not be used in learning weights in the network in any way. We will only use it to evaluate if our model is overfitting and select a stopping point for training.
# 
# One other thing: we have to reshape the vector of `X_train` and `X_test` into a matrix. We can use the `keras.preprocessing.sequence.pad_sequences` to pad the shorter sequences with an extra code (in our case, 26, given that the original coding is from 0-25).
# 
# Note that we convert the matrix to float 32 for the benefit of running it on GPU.

# In[17]:

max_len = 14
# add one padding feature
max_features = len(IUPAC.extended_protein.letters) + 1 

train_idx = meta[(meta["set"] == "train") & 
                 (meta["mhc"] == "HLA-A*02:01") & 
                 (meta["peptide_length"] <= max_len)].index.values
test_idx = meta[(meta["set"] == "test") &
                (meta["mhc"] == "HLA-A*02:01") & 
                (meta["peptide_length"] <= max_len)].index.values

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

X_train = sequence.pad_sequences(X_train, maxlen=max_len, 
                                 value=max_features-1)
X_test = sequence.pad_sequences(X_test, maxlen=max_len, 
                                value=max_features-1)

w_train = sample_weights[train_idx].astype(np.float32)
w_test = sample_weights[test_idx].astype(np.float32)


# ## Build a model

# Now let's build a deep learning model!

# ### Set model parameters
# 
# Let's first define some parameters

# In[18]:

batch_size = 128 
nb_epoch = 10 # only train for 10 iteration

embedding_size = 32
num_conv_filters = 32 # number of convolutional filters
conv_filter_size = 3 # size of the convolutional filters
num_lstm_units = 64 # number of LSTM units


# In[23]:

model = Sequential()

# 4 layers of conv
model.add(Embedding(max_features, embedding_size, input_length=max_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters=num_conv_filters,
                 kernel_size=conv_filter_size))
model.add(BatchNormalization(axis=1))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))

model.add(LSTM(num_lstm_units))
model.add(Dense(1, kernel_regularizer=l1_l2(l1=1e-2, l2=1e-2),
                activation="relu"))


# You can access the number of parameters and the shape of output in each layer by the `.summary()` function.

# In[24]:

model.summary()


# ### Embedding layers
# 
# The embedding layers are often used in NLP (such as this [sentimental analysis from imdb](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py)). It turns the indexed input, in our case the encoded amino acid sequences, into a dense vector of fixed size, _e.g._ [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]
# 
# The parameters of the embedding layers try to capture the relationships between the encoded elements, for example, the similarities between amino acids or the similar meanings across words. Elements with high similarities will have similar vector representations. 
# 
# The number of parameters of an embedding layer is the `dimension_of_codebook` multiplied by the `number of embedding units`. In our case, it's 
# 
# $$ 26 \text{ alphabets in the codebook} \times 32 \text{ embedding units} = 832 \text{ parameters}$$

# ### Dropout layer
# 
# Large number of parameters tends to overfit the training data, the number of parameters in deep neural network is gigantic. One simple technique that is offen used to prevent overfitting is to insert a [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) layer in between fully connected layers (_e.g._ LSTM). A dropout layer randomly set a fraction `p` of input units to 0 at each update during training time: 
# 
# ![](../figures/dropout.png)

# ### Convolutional layers
# 
# The convolutional layer has a sparse connection to the adjecent units:
# 
# ![](http://deeplearning.net/tutorial/_images/conv_1D_nn.png)
# 
# In the figure above, lines with same color correspond to the same weight. 
# 
# For activation, the convolution layers use Rectified Linear Unit (ReLU), which is shown to have [better convergence](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) than `tanh` function.
# 
# ![](http://cs231n.github.io/assets/nn1/relu.jpeg)
# 
# You can see the first convolutional layer contains 
# 
# $$  32 \text{ embedding units from last layer} \times 32 \text{ convolutional units} \times 3 \text{ (convolutional filter size)} + 32 \text{ bias} = 3104 \text{ parameters}$$

# ### Batch normalization layer
# 
# According to [this paper](https://arxiv.org/pdf/1502.03167v3.pdf) from Google, a batch normalization layer can reduce internal covariate shift between layers and thus reduce the convergence time and might eliminatethe need for Dropout. Each unit in batch normailzation calculates the mean and standard deviation in the mini batch and shift the input by
# 
# $$ y = \gamma x + \beta$$
# 
# Therefore, the unmber of parameter is $2 \times 14 = 28$ in the first BatchNormalization layer.

# ### Max pooling layer
# 
# A max pooling layer parttions input into non-overlapping rectangles and output the maximum in each subregion. It reduces the data size and thus speed up the computation. Usually the pooling size is 2, too much pooling may result in loss of too much information. 

# ### LSTM layer
# 
# Long short-term memory (LSTM) layer is one of the most widely used recurrent neural network (RNN) layer. The units in the recurrent neural network forms directed cycles which allow them to exhibit temporal behavior. Therefore we often see the application of RNNs in sequential data such as [handwriting recognition](https://arxiv.org/pdf/1312.4569.pdf), [speech recognition](https://www.microsoft.com/en-us/research/publication/lstm-time-and-frequency-recurrence-for-automatic-speech-recognition/), or [sentimental analysis](http://deeplearning.net/tutorial/lstm.html).
# 
# A LSTM unit is different from other RNN unit in the sense that it has a forget gate:
# 
# ![](http://deeplearning.net/tutorial/_images/lstm_memorycell.png)
# 
# For each of the unit at time $t$, it involves the following calculation:
# 
#   * Input gate 
#       $$i_t = \sigma(W_ix_t + U_i h_{t-1} + b_i)$$
#   
#   * Candidate value for the state of memory cell 
#       $$\tilde{C}_t = tanh(W_cx_t + U_ch_{t-1} + b_c)$$
#   
#   * Activation of forget cell
#       $$f_t = \sigma(W_fx_t + U_fh_{t-1} + b_f)$$
#   
#   * The state of memory cell 
#       $$C_t = i_t \tilde{C}_t + f_t C_{t-1}$$
#   
#   * Output gate
#       $$o_t = \sigma(W_ox_t + U_o h_{t-1} + b_o)$$
#   
#   * Output 
#       $$h_t = o_t tanh(C_t)$$
# 
# The above calculations involve the following parameters:
#   * $W$ : weight vector of `input_size`
#   * $U$ : weight vector of `output_size`
#   * $b$ : a scalar of bias
# 
# Therefore, for instance, the LSTM layer involves
# 
# $$(64 \text{ units of LSTM layer} + 32 \text{ units of previous layer} + 1 \text{ bias}) \times 64 \text{ units} \times 4 (\text{input gate, candidate state, forget, output gate}) = 24832 \text{ parameters} $$

# ## Configure model for training
# 
# With the model structure in place, we will configure the training methods for the model.

# In[25]:

model.compile(loss="mean_absolute_error",
              optimizer="rmsprop")


# ### Choose optimizer
# 
# Optimizers are basically a gradient descent algorithm used for optimizing the weights during each update. Here we will use the `RMSProp` optimizer as it is suggested to be [a good choice for recurrent neural networks](https://keras.io/optimizers/#rmsprop). A list of other optimizers and their formulation can be found in [this](http://sebastianruder.com/optimizing-gradient-descent/index.html)  great blog post. 

# ### `compile`
# 
# Although the name might be misleading, this function does not really start compiling the model into C++ code, instead just setting more training behavior. Here we use the categorical [crossentropy](https://en.wikipedia.org/wiki/Cross_entropy) as the target value to optimize. In each iteration (or as the deep learning guys like to call it, the `epoch`) we will also output the prediction accuracy as output. 

# ## Finally ... we start training

# And now we can train

# In[26]:

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    validation_data=(X_test, y_test),
                    sample_weight=w_train,
                    shuffle=True)


# We can make predictions for new data by doing:

# In[27]:

y_test_pred = model.predict(X_test)


# There are two ways of evaluating the performance of the model
# 
# 1. A general practice in terms of binding affinity is to use IC50 < 500 as high affinity. We can use this threshold to calculate the accuracy.
# 2. We can also calculate the Spearman correlation between the observed affinity with the predicted one.

# In[28]:

auc = roc_auc_score(y_test > np.log10(500), y_test_pred)
spearman = stats.spearmanr(y_test, y_test_pred)

print("AUROC: {:.4f}".format(auc))
print("SpearmanR: {:.4f}".format(spearman.correlation))


# ## More
# 
# For other techniques for training model you may go through the following links:
# 
# * [IEDB automated server benchmark](http://tools.iedb.org/auto_bench/mhci/weekly/) and [the paper](https://www.ncbi.nlm.nih.gov/pubmed/25717196)
# * [`mhcflurry`](https://github.com/hammerlab/mhcflurry) is an open-source project by [Jeff Hammerbacher's lab](http://www.hammerlab.org/). It's a great source to learn about building deep learning models for MHC binding prediction, but the code is not very straitforward ...
# 
# 
# * Callback functions: [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint) or [`EarlyStopping`](https://keras.io/callbacks/#earlystopping) are quite useful
# * Hyperparameter search using `sklearn`'s [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) function. An example can be found in `kera`'s [example script](https://github.com/fchollet/keras/blob/master/examples/mnist_sklearn_wrapper.py).
# * Parallelization: [`mxnet`](https://github.com/dmlc/mxnet) or [Spark](https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html)
# * An nice overview on available deep learning library out there (more focused on python): [My Top 9 Favorite Python Deep Learning Libraries](http://www.pyimagesearch.com/2016/06/27/my-top-9-favorite-python-deep-learning-libraries/)
