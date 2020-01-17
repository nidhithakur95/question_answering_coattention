
# %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib
import sys
import os
import zipfile
import tarfile
import json 
import hashlib
import re
import itertools
from tensorflow.contrib.layers import xavier_initializer
import logging

glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.50d.txt"

# 15 MB
data_set_zip = "tasks_1-20_v1-2.tar.gz"

#Select "task 5"
train_set_file = "qa5_three-arg-relations_train.txt"
test_set_file = "qa5_three-arg-relations_test.txt"

train_set_post_file = "tasks_1-20_v1-2/en/"+train_set_file
test_set_post_file = "tasks_1-20_v1-2/en/"+test_set_file

try: from urllib.request import urlretrieve, urlopen
except ImportError: 
    from urllib import urlretrieve
    from urllib2 import urlopen
#large file - 862 MB
if (not os.path.isfile(glove_zip_file) and
    not os.path.isfile(glove_vectors_file)):
    urlretrieve ("http://nlp.stanford.edu/data/glove.6B.zip", 
                 glove_zip_file)
if (not os.path.isfile(data_set_zip) and
    not (os.path.isfile(train_set_file) and os.path.isfile(test_set_file))):
    urlretrieve ("https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz", 
                 data_set_zip)

def unzip_single_file(zip_file_name, output_file_name):
    """
        If the output file is already created, don't recreate
        If the output file does not exist, create it from the zipFile
    """
    if not os.path.isfile(output_file_name):
        with open(output_file_name, 'wb') as out_file:
            with zipfile.ZipFile(zip_file_name) as zipped:
                for info in zipped.infolist():
                    if output_file_name in info.filename:
                        with zipped.open(info) as requested_file:
                            out_file.write(requested_file.read())
                            return
def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
    if not os.path.isfile(output_file_name):
        with tarfile.open(zip_file_name) as un_zipped:
            un_zipped.extract(interior_relative_path+output_file_name)    
unzip_single_file(glove_zip_file, glove_vectors_file)
targz_unzip_single_file(data_set_zip, train_set_file, "tasks_1-20_v1-2/en/")
targz_unzip_single_file(data_set_zip, test_set_file, "tasks_1-20_v1-2/en/")

# Deserialize GloVe vectors
glove_wordmap = {}
with open(glove_vectors_file, "r", encoding="utf8") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")

wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

# Gather the distribution hyperparameters
v = np.var(s,0) 
m = np.mean(s,0) 
RS = np.random.RandomState()

def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(m,np.diag(v))
    return glove_wordmap[unk]

def sentence2sequence(sentence):
    """
    - Turns an input paragraph into an (m,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
      TensorFlow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      TensorFlow provides. Normal Python suffices for this task.
    """
    tokens = sentence.strip('"(),-').lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
                continue
            else:
                i = i-1
            if i == 0:
                # word OOV
                # https://arxiv.org/pdf/1611.01436.pdf
                rows.append(fill_unk(token))
                words.append(token)
                break
    return np.array(rows), words

def contextualize(set_file):
    """
    Read in the dataset of questions and build question+answer -> context sets.
    Output is a list of data points, each of which is a 7-element tuple containing:
        The sentences in the context in vectorized form.
        The sentences in the context as a list of string tokens.
        The question in vectorized form.
        The question as a list of string tokens.
        The answer in vectorized form.
        The answer as a list of string tokens.
        A list of numbers for supporting statements, which is currently unused.
    """
    data = []
    context = []
    with open(set_file, "r", encoding="utf8") as train:
        for line in train:
            l, ine = tuple(line.split(" ", 1))
            # Split the line numbers from the sentences they refer to.
            if l is "1":
                # New contexts always start with 1, 
                # so this is a signal to reset the context.
                context = []
            if "\t" in ine: 
                # Tabs are the separator between questions and answers,
                # and are not present in context statements.
                question, answer, support = tuple(ine.split("\t"))
                data.append((tuple(zip(*context))+
                             sentence2sequence(question)+
                             sentence2sequence(answer)+
                             ([int(s) for s in support.split()],)))
                # Multiple questions may refer to the same context, so we don't reset it.
            else:
                # Context sentence.
                context.append(sentence2sequence(ine[:-1]))
    return data
train_data = contextualize(train_set_post_file)
test_data = contextualize(test_set_post_file)

final_train_data = []
def finalize(data):
    """
    Prepares data generated by contextualize() for use in the network.
    """
    final_data = []
    for cqas in train_data:
        contextvs, contextws, qvs, qws, avs, aws, spt = cqas

        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words = sum(contextws,[])

        # Location markers for the beginnings of new sentences.
        sentence_ends = np.array(list(lengths)) 
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
    return np.array(final_data)
final_train_data = finalize(train_data)   
final_test_data = finalize(test_data)

tf.reset_default_graph()
# Hyperparameters

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 64

# The number of dimensions in our word vectorizations.
D = 100 

# How quickly the network learns. Too high, and we may run into numeric instability 
# or other issues.
learning_rate = 0.001

# Dropout probabilities. For a description of dropout and what these probabilities are, 
# see Entailment with TensorFlow.
input_p, output_p = 0.5, 0.5

# How many questions we train on at a time.
batch_size = 64

# Number of passes in episodic memory. We'll get to this later.
passes = 4

# Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
ff_hidden_size = 256

weight_decay = 0.00000001
# The strength of our regularization. Increase to encourage sparsity in episodic memory, 
# but makes training slower. Don't make this larger than leraning_rate.

training_iterations_count = 100000
# How many questions the network trains on each time it is trained. 
# Some questions are counted multiple times.

display_step = 100
# How many iterations of training occur before each validation check.

max_q_length = 30  
max_c_length = 400


#need to make change here
apply_dropout = False
dropout_encoder = 0.7 



#rnn state size
rnn_state_size = 100

l2 = 0.01


#adding placeholders
q_input_placeholder = tf.placeholder(tf.int32, (None, max_q_length), name="q_input_ph")
q_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, max_q_length),name="q_mask_placeholder")
c_input_placeholder = tf.placeholder(tf.int32, (None, max_c_length), name="c_input_ph")
c_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, max_c_length),name="c_mask_placeholder")
labels_placeholderS = tf.placeholder(tf.int32, (None, max_c_length), name="label_phS")
labels_placeholderE = tf.placeholder(tf.int32, (None, max_c_length), name="label_phE")
dropout_placeholder = tf.placeholder(tf.float32, name="dropout_ph")




# Input Module

# Context: A [batch_size, maximum_context_length, word_vectorization_dimensions] tensor 
# that contains all the context information.
context = tf.placeholder(tf.float32, [None, None, D], "context")  
context_placeholder = context # I use context as a variable name later on

# input_sentence_endings: A [batch_size, maximum_sentence_count, 2] tensor that 
# contains the locations of the ends of sentences. 
input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

# recurrent_cell_size: the number of hidden units in recurrent layers.
input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

# input_p: The probability of maintaining a specific hidden input unit.
# Likewise, output_p is the probability of maintaining a specific hidden output unit.
gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)

# dynamic_rnn also returns the final internal state. We don't need that, and can
# ignore the corresponding output (_). 
input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope = "input_module")

# cs: the facts gathered from the context.
cs = tf.gather_nd(input_module_outputs, input_sentence_endings)
# to use every word as a fact, useful for tasks with one-sentence contexts
print(cs)
s = input_module_outputs
#print(cs)




# Question Module

# query: A [batch_size, maximum_question_length, word_vectorization_dimensions] tensor 
#  that contains all of the questions.

query = tf.placeholder(tf.float32, [None, None, D], "query")

# input_query_lengths: A [batch_size, 2] tensor that contains question length information. 
# input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range() 
# so that it plays nice with gather_nd.
input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32, 
                                               scope = tf.VariableScope(True, "input_module"))

# q: the question states. A [batch_size, recurrent_cell_size] tensor.
q = tf.gather_nd(question_module_outputs, input_query_lengths)
print(q)

#Episodic Memory


#Question vector
Qprime = question_module_outputs
q_senti = tf.get_variable("q_senti0", (recurrent_cell_size,), dtype=tf.float32)
q_senti = tf.tile(q_senti, tf.shape(Qprime)[0:1])
q_senti = tf.reshape(q_senti, (-1, 1, tf.shape(Qprime)[2]))
print(Qprime)
print(q_senti)
Qprime = tf.concat([Qprime, q_senti], axis=1)
Qprime = tf.transpose(Qprime, [0, 2, 1], name="Qprime")
#print(max_q_length)
WQ = tf.get_variable("WQ", (max_q_length + 1, max_q_length + 1),initializer=tf.contrib.layers.xavier_initializer())
print(WQ)
bQ = tf.get_variable("bQ_Bias", shape=(recurrent_cell_size, max_q_length + 1),
                             initializer=tf.contrib.layers.xavier_initializer())
print(bQ)

Q = tf.einsum('ijk,kl->ijl', Qprime, WQ)
Q = tf.nn.tanh(Q + bQ, name="Q")

print(Q)
enc_keep_prob = tf.maximum(tf.constant(dropout_encoder),0.5)


#input Vector
D = input_module_outputs
print(D)
c_senti = tf.get_variable("c_senti0", (recurrent_cell_size,), dtype=tf.float32)
c_senti = tf.tile(c_senti, tf.shape(D)[0:1])
c_senti = tf.reshape(c_senti, (-1, 1, tf.shape(D)[2]))
print(c_senti)
D = tf.concat([D, c_senti], axis=1)




#transpose andd all
D = tf.transpose(D, [0, 2, 1])
L = tf.einsum('ijk,ijl->ikl', D, Q)
AQ = tf.nn.softmax(L)
AD = tf.nn.softmax(tf.transpose(L, [0, 2, 1]))
CQ = tf.matmul(D, AQ)
CD1 = tf.matmul(Q, AD)
CD2 = tf.matmul(CQ, AD)
CD = tf.concat([CD1, CD2], axis=1)
CDprime = tf.concat([CD, D], axis=1)
CDprime = tf.transpose(CDprime, [0, 2, 1])

print(CDprime)

with tf.variable_scope("u_rnn", reuse=False):
            cell_fw = tf.contrib.rnn.GRUCell(64)
            cell_bw = tf.contrib.rnn.GRUCell(64)
            if apply_dropout:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=enc_keep_prob)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=enc_keep_prob)

            (cc_fw, cc_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=CDprime,dtype=tf.float32)

U = tf.concat([cc_fw, cc_bw], axis=2)
logging.debug("U={}".format(U))
print(U)

#Answer Module - Decoder section

apply_dropout=True
apply_l2_reg=False
pool_size=4
cumulative_loss=True
float_mask = tf.cast(c_mask_placeholder, dtype=tf.float32)
neg = tf.constant([0], dtype=tf.float32)
neg = tf.tile(neg, [tf.shape(float_mask)[0]])
neg = tf.reshape(neg, (tf.shape(float_mask)[0], 1))
float_mask = tf.concat([float_mask, neg], axis=1)
labels_S = tf.concat([labels_placeholderS, tf.cast(neg, tf.int32)], axis=1)
labels_E = tf.concat([labels_placeholderE, tf.cast(neg, tf.int32)], axis=1)
dim = recurrent_cell_size


# initialize us and ue as first word in context
i_start = tf.zeros(shape=(tf.shape(U)[0],), dtype='int32')

i_end = tf.zeros(shape=(tf.shape(U)[0],), dtype='int32')
idx = tf.range(0, tf.shape(U)[0], 1)
s_idx = tf.stack([idx, i_start], axis=1)
e_idx = tf.stack([idx, i_end], axis=1)
us = tf.gather_nd(U, s_idx) 
print(us)
ue = tf.gather_nd(U, e_idx)
print(ue)




def HMN_func(dim, ps):  # ps=pool size, HMN = highway maxout network
            def func(ut, h, us, ue):
                h_us_ue = tf.concat([h, us, ue], axis=1)
                WD = tf.get_variable(name="WD", shape=(5 * dim, dim), dtype='float32',
                                     initializer=xavier_initializer())
                r = tf.nn.tanh(tf.matmul(h_us_ue, WD))
                ut_r = tf.concat([ut, r], axis=1)
                if apply_dropout:
                    ut_r = tf.nn.dropout(ut_r, keep_prob=dropout_placeholder)
                W1 = tf.get_variable(name="W1", shape=(3 * dim, dim, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b1 = tf.get_variable(name="b1_Bias", shape=(dim, ps), dtype='float32',
                                     initializer=tf.zeros_initializer())
                mt1 = tf.einsum('bt,top->bop', ut_r, W1) + b1
                mt1 = tf.reduce_max(mt1, axis=2)
                if apply_dropout:
                    mt1 = tf.nn.dropout(mt1, dropout_placeholder)
                W2 = tf.get_variable(name="W2", shape=(dim, dim, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b2 = tf.get_variable(name="b2_Bias", shape=(dim, ps), dtype='float32',
                                     initializer=tf.zeros_initializer())
                mt2 = tf.einsum('bi,ijp->bjp', mt1, W2) + b2
                mt2 = tf.reduce_max(mt2, axis=2)
                mt12 = tf.concat([mt1, mt2], axis=1)
                if apply_dropout:
                    mt12 = tf.nn.dropout(mt12, keep_prob=dropout_placeholder)
                W3 = tf.get_variable(name="W3", shape=(2 * dim, 1, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b3 = tf.get_variable(name="b3_Bias", shape=(1, ps), dtype='float32', initializer=tf.zeros_initializer())
                hmn = tf.einsum('bi,ijp->bjp', mt12, W3) + b3
                hmn = tf.reduce_max(hmn, axis=2)
                hmn = tf.reshape(hmn, [-1])
                return hmn

            return func

HMN_alpha = HMN_func(dim, pool_size)
print(HMN_alpha)
HMN_beta = HMN_func(dim, pool_size)
print(HMN_beta)

alphas, betas = [], []
h = tf.zeros(shape=(tf.shape(U)[0], dim), dtype='float32', name="h_dpd") 
print(h)
U_transpose = tf.transpose(U, [1, 0, 2])
print(U_transpose)

with tf.variable_scope("dpd_RNN"):
            cell = tf.contrib.rnn.GRUCell(dim)
            for time_step in range(3):  # number of time steps can be considered as a hyper parameter
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()

                us_ue = tf.concat([us, ue], axis=1)
                
                print(us_ue)
                _, h = cell(inputs=us_ue, state=h)
                
                print(h)

                with tf.variable_scope("alpha_HMN"):
                    if time_step >= 1:
                        tf.get_variable_scope().reuse_variables()
                    alpha = tf.map_fn(lambda ut: HMN_alpha(ut, h, us, ue), U_transpose, dtype=tf.float32)
                    alpha = tf.transpose(alpha, [1, 0]) * float_mask

    
                i_start = tf.argmax(alpha, 1)
                idx = tf.range(0, tf.shape(U)[0], 1)
                s_idx = tf.stack([idx, tf.cast(i_start, 'int32')], axis=1)
                us = tf.gather_nd(U, s_idx)

                with tf.variable_scope("beta_HMN"):
                    if time_step >= 1:
                        tf.get_variable_scope().reuse_variables()
                    beta = tf.map_fn(lambda ut: HMN_beta(ut, h, us, ue), U_transpose, dtype=tf.float32)
                    beta = tf.transpose(beta, [1, 0]) * float_mask

                i_end = tf.argmax(beta, 1)
                e_idx = tf.stack([idx, tf.cast(i_end, 'int32')], axis=1)
                ue = tf.gather_nd(U, e_idx)

                alphas.append(alpha)
                betas.append(beta)

if cumulative_loss:
    losses_alpha = [tf.nn.softmax_cross_entropy_with_logits(labels=labels_S, logits=a) for a in
                            alphas]
    losses_alpha = [tf.reduce_mean(x) for x in losses_alpha]
    losses_beta = [tf.nn.softmax_cross_entropy_with_logits(labels=labels_E, logits=b) for b in
                           betas]
    losses_beta = [tf.reduce_mean(x) for x in losses_beta]

    loss = tf.reduce_sum([losses_alpha, losses_beta])
else:
    cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=labels_S, logits=alpha,
                                                                          name="cross_entropy_start")
    cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=labels_E, logits=beta,
                                                                        name="cross_entropy_end")
    loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

if apply_l2_reg:
    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "Bias" not in v.name])
    loss += loss_l2 * l2_lambda
