
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from math import sqrt
import matplotlib
import numpy
from numpy import concatenate
import matplotlib.pyplot as plt
sys.path.append(str(Path(os.path.abspath('..')).parent)+'/secret')
import gensim
import tensorflow as tf
from keras import backend as K
import codecs
import re
num_cores = 4
'''
num_CPU = 1
num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.compat.v1.Session(config=config)
K.set_session(session)
'''
path = '../../secret/data/'
'''
df = pd.read_csv(path+'trainingset/vec/dru.csv',index_col=0)
df = df[['txn','drug','icd10']]
df = df.fillna(0)
scaler = MinMaxScaler()
df[['drug', 'icd10']] = scaler.fit_transform(df[['drug', 'icd10']])
df['drug_1'] = df.groupby(['txn'])['drug'].shift(1)
df['icd10_1'] = df.groupby(['txn'])['icd10'].shift(1)
df = df[~np.isnan(df['drug_1'])]
df = df[df['drug']!=df['drug_1']]
print(df)

# split into train and test sets
train, test = df[0:700000], df[700000:len(df)]
print(len(train), len(test))

trainX = train[['drug']].to_numpy()
trainY = train[['drug_1']].to_numpy()
testX = test[['drug']].to_numpy()
testY = test[['drug_1']].to_numpy()
time_steps = 1
n_features = 1
trainX = numpy.reshape(trainX, (trainX.shape[0], time_steps, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], time_steps, trainX.shape[1]))
with tf.device('/cpu:0'):
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, activation='softmax'))
        model.add(LSTM(512, activation='softmax'))
        model.add(Dense(1, activation='softmax'))
        #optimizer = optimizers.Adam(clipvalue=0.5)
        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer, loss='mse')

        history = model.fit(trainX, trainY, epochs=20, batch_size=100, verbose=1)
        plt.plot(history.history['loss'], label="loss")
        plt.legend(loc="upper right")
        plt.show()
'''
def save_file(df,p):
	file = Path(p)
	if file.is_file():
		with open(p, 'a', encoding="utf-8") as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(p)

def remove_file(p):
	file = Path(p)
	if file.is_file():
		os.remove(p)

def chain(max,df,n1,n2):
        toggle = True
        d = df.sample(n=1)
        # d = df.sample(n=1,weights=df[n1+'_weight'])
        last = d[n1].iloc[0]
        text = []
        n = 0
        for i in range(max):
                if toggle:
                        text.append([last, last])
                        x = [last]
                        y = [last]
                        icd10 = df[df[n1] == last].sample(n=1)[n2].iloc[0]
                        x.insert(0, icd10)
                        last = df[df[n2] == icd10].sample(n=1)[n1].iloc[0]
                        y.insert(0, last)
                        text.append(x)
                        text.append(y)
                        last = icd10
                        toggle = False
                else:
                        text.append([last, last])
                        x = [last]
                        y = [last]
                        drug = df[df[n2] == last].sample(n=1)[n1].iloc[0]
                        x.insert(0, drug)
                        last = df[df[n1] == drug].sample(n=1)[n2].iloc[0]
                        y.insert(0, last)
                        text.append(x)
                        text.append(y)
                        last = drug
                        toggle = True

        return text

def chain2(max,df,n1,n2):
        toggle = True
        d = df.sample(n=1)
        #d = df.sample(n=1,weights=df[n1+'_weight'])
        last = d[n1].iloc[0]
        text = []
        n = 0
        for i in range(max):
                if toggle:
                        text.append([last,last])
                        x = [last]
                        y = [last]
                        icd10 = df[df[n1] == last].sample(n=1, weights=df[n2 + '_weight'])[n2].iloc[0]
                        x.insert(0,icd10)
                        last = df[df[n2] == icd10].sample(n=1, weights=df[n1 + '_weight'])[n1].iloc[0]
                        y.insert(0,last)
                        text.append(x)
                        text.append(y)
                        last = icd10
                        toggle = False
                else:
                        text.append([last, last])
                        x = [last]
                        y = [last]
                        drug = df[df[n2] == last].sample(n=1, weights=df[n1 + '_weight'])[n1].iloc[0]
                        x.insert(0,drug)
                        last = df[df[n1] == drug].sample(n=1, weights=df[n2 + '_weight'])[n2].iloc[0]
                        y.insert(0,last)
                        text.append(x)
                        text.append(y)
                        last = drug
                        toggle = True

        return text

'''
df = pd.read_csv(path+'testset/raw/dru.csv',index_col=0)
#df = df[['drug','icd10']]
df['icd10_weight'] = df.groupby(['icd10'])['drug'].transform('count')
df['icd10_weight'] = 1-(df['icd10_weight']/df['icd10_weight'].max())
df['drug_weight'] = df.groupby(['drug'])['icd10'].transform('count')
df['drug_weight'] = 1-(df['drug_weight']/df['drug_weight'].max())
df = df.reset_index()
df = df[df['icd10']=='J029']
df = df.sort_values(by=['drug_weight'], ascending=[True])
print(df.head(50))
'''
'''
df = pd.read_csv(path+'testset/raw/dru.csv',index_col=0)
df['icd10_weight'] = df.groupby(['icd10'])['drug'].transform('count')
df['icd10_weight'] = 1-(df['icd10_weight']/df['icd10_weight'].max())
df['drug_weight'] = df.groupby(['drug'])['icd10'].transform('count')
df['drug_weight'] = 1-(df['drug_weight']/df['drug_weight'].max())
df = df.reset_index()
while True:
        text = chain2(100, df, 'drug', 'icd10')
        print(text)
'''
def train_chain1(filename,modelname):
        #chain 1 model without weight
        df = pd.read_csv(path+'trainingset/raw/'+filename+'.csv',index_col=0)
        df = df[['drug','icd10']]

        t = 1000000
        p = 100
        s = 0
        model = None
        for i in range(int(t/p),-1,-1*p):
                file = Path(path+modelname+ '_'+str(i))
                if file.is_file():
                        model = gensim.models.Word2Vec.load(path + modelname + '_'+str(i))
                        s = i
                        break
        if model is None:
                model = gensim.models.Word2Vec(chain(1000,df,'drug','icd10'), compute_loss = True, sg = 1)
        print(s)
        for i in range(s+1,t):
                text = chain(100,df,'drug','icd10')
                #model = gensim.models.Word2Vec.load(path + modelname)
                model.build_vocab(text, update=True)
                model.train(text, total_examples=model.corpus_count, compute_loss = True, epochs=10)
                print(i)
                print(model.get_latest_training_loss())
                if i % p == 0:
                        model.save(path+modelname+'_'+str(i))
                        print('saved '+modelname+'_'+str(i))

'''
#chain 2 model2 with weight
df = pd.read_csv(path+'trainingset/raw/dru.csv',index_col=0)
df = df[['drug','icd10']]
df['icd10_weight'] = df.groupby(['icd10'])['drug'].transform('count')
df['icd10_weight'] = 1-(df['icd10_weight']/df['icd10_weight'].max())
df['drug_weight'] = df.groupby(['drug'])['icd10'].transform('count')
df['drug_weight'] = 1-(df['drug_weight']/df['drug_weight'].max())
df = df.reset_index()

#model = gensim.models.Word2Vec(chain2(1000,df,'drug','icd10'), compute_loss = True, sg = 1)
#model.save(path+'model2')

for i in range(100000):
        text = chain2(100,df,'drug','icd10')
        model = gensim.models.Word2Vec.load(path + 'model2')
        model.build_vocab(text, update=True)
        model.train(text, total_examples=model.corpus_count, compute_loss = True, epochs=10)
        print(model.get_latest_training_loss())
        model.save(path+'model2')
'''

'''
df['drug_name'] = df['drug_name'].str.strip()
df['drug_1'] = df.groupby(['txn'])['drug'].shift(1)
df['icd10_1'] = df.groupby(['txn'])['icd10'].shift(1)
df['drug_2'] = df.groupby(['txn'])['drug'].shift(2)
df['icd10_2'] = df.groupby(['txn'])['icd10'].shift(2)
df['drug_3'] = df.groupby(['txn'])['drug'].shift(3)
df['icd10_3'] = df.groupby(['txn'])['icd10'].shift(3)
'''
'''
df = pd.read_csv(path+'trainingset/raw/dru.csv',index_col=0)
df = df[['drug','icd10']]
#df = df[['drug','icd10']]
df2 = df.apply(lambda x: ','.join(x.astype(str)), axis=1)
df_clean = pd.DataFrame({'clean': df2})
sent = [row.split(',') for row in df_clean['clean']]
#df = df[['drug','icd10']]
print(sent)
model = gensim.models.Word2Vec(sent, sg = 1)
model.save(path+'test_model')
'''

def test_chain(prefix=''):
        df = pd.read_csv(path+'testset/raw/'+prefix+'dru.csv',index_col=0)
        df['drug_name'] = df['drug_name'].str.strip()
        drug_map = dict(zip(df['drug'], df['drug_name']))
        model = gensim.models.Word2Vec.load(path+prefix+'model')
        #print(model.wv.vocab)
        icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
        icd10_map = dict(zip(icd10['code'], icd10['cdesc']))
        x = 'J029' #Acute pharyngitis, unspecified
        y = 'AMOT08' #Amoxicillin cap 500 mg
        y = 'BERT01' #Beramol Tab  500  mg
        y = 'DOST02' #,*(d) Dosanac Tab 50 mg,1,N200
        x = 'M0699' #Rheumatoid arthritis, unspecified Site unspecified
        #x = 'C795' #Secondary malignant neoplasm of bone and bone marrow
        #x = 'R509' #Fever, unspecified
        #x = 'E104'
        y = 'DOUT01'
        #x = 'M8195'
        y = 'PROI11'
        x = 'Z94' #Transplanted organ and tissue status
        x = 'K250' #Gastric ulcer: acutewith haemorrhage
        x = 'K226'#K226,Gastro-oesophageal laceration-haemorrhage syndrome
        x = 'K590' #K21,Gastro-oesophageal reflux disease
        #x = 'L209'#L209,"Atopic dermatitis, unspecified
        #x = 'J029'  # Acute pharyngitis, unspecified
        #x = 'H811' #BPPV
        x = 'N179'#N179  Acute renal failure, unspecified
        x = 'M329'#  Systemic lupus erythematosus, unspecified
        x = 'C910'#  Acute lymphoblastic leukaemia
        similar_words = model.wv.most_similar(positive=[x],topn=100)
        #print(icd10_map['J029'])
        if x in icd10_map:
                print(icd10_map[x])
        if x in drug_map:
                print(drug_map[x])
        for i in range(len(similar_words)):
                if similar_words[i][0] in icd10_map:
                        #print(str(similar_words[i][1])+' '+icd10_map[similar_words[i][0]])
                        pass
                if similar_words[i][0] in drug_map:
                        print(str(similar_words[i][1])+' '+drug_map[similar_words[i][0]])

def validate(model_list, prefix='', save=False, n=None):
        df = pd.read_csv(path+'trainingset/raw/'+prefix+'dru.csv',index_col=0)
        df['drug_name'] = df['drug_name'].str.strip()
        drug_map = dict(zip(df['drug'], df['drug_name']))
        #model = gensim.models.Word2Vec.load(path+'model')
        icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
        icd10['cdesc'] = icd10['cdesc'].str.strip()
        icd10_map = dict(zip(icd10['code'], icd10['cdesc']))

        #print(len(drug_map))
        #print(len(model.wv.vocab))
        df = pd.read_csv(path+'testset/raw/'+prefix+'dru.csv',index_col=0)
        if n is not None:
                df = df.head(n)
        df['drug_name'] = df['drug_name'].str.strip()
        df['icd10_name'] = df['icd10'].map(icd10_map)
        df['icd10_name'] = df['icd10_name'].str.strip()
        #remove_file(path+'validation.csv')
        #remove_file(path+prefix+'validation.txt')
        for m in model_list:
                model = gensim.models.Word2Vec.load(path+m)
                data = []
                for txn in df['txn'].unique().tolist():
                        d = df[df['txn']==txn]
                        #print(d)
                        #p = []
                        #dx = []
                        #for x in d['icd10'].unique().tolist():
                        #        if x in model.wv.vocab and x in icd10_map:
                        #                dx.append(x)

                        #rank = []
                        #top5 = []
                        '''
                        #if len(dx)>0:
                                
                                #Approach 1: calcualte similar wards from all dx at once
                                similar_words = model.wv.most_similar(positive=dx, topn=5000)
                                n = 0
                                for i in range(len(similar_words)):
                                        da = [txn]
                                        if similar_words[i][0] in drug_map:
                                                da.append(similar_words[i][0])
                                                da.append(drug_map[similar_words[i][0]])
                                                da.append(similar_words[i][1])
                                                p.append(similar_words[i][0])
                                                #print(str(similar_words[i][1]) + ' ' + drug_map[similar_words[i][0]])
                                                data.append(da)
                                                if similar_words[i][0] in d['drug'].unique().tolist():
                                                        rank.append([n,drug_map[similar_words[i][0]]])
                                                if len(top5) < 5:
                                                        top5.append([n, drug_map[similar_words[i][0]]])
                                                n = n+1
                                
                                
                                #txt = ''
                                #txt = txt + '#####################' + '\n'
                                #txt = txt + str(d['icd10_name'].unique().tolist()) + '\n\n'
                                #for dd in dx:
                        '''
                        # Approach 2: get each dx and treat the rest of dx is negative
                        dx = d['icd10'].unique().tolist()
                        for x in dx:
                                if x in model.wv.vocab and x in icd10_map:
                                        rank = []
                                        top5 = []
                                        neg = dx.copy().remove(x)
                                        similar_words = model.wv.most_similar(positive=[x], negative=neg, topn=len(model.wv.vocab))
                                        detected_drug = []
                                        n = 1
                                        for i in range(len(similar_words)):
                                                #da = [txn]
                                                if similar_words[i][0] in drug_map:
                                                        #da.append(similar_words[i][0])
                                                        #da.append(drug_map[similar_words[i][0]])
                                                        #da.append(similar_words[i][1])
                                                        #p.append(similar_words[i][0])
                                                        # print(str(similar_words[i][1]) + ' ' + drug_map[similar_words[i][0]])
                                                        #data.append(da)
                                                        if similar_words[i][0] in d['drug'].unique().tolist():
                                                                detected_drug.append(similar_words[i][0])
                                                                #rank.append([n, drug_map[similar_words[i][0]]])
                                                                data.append([txn,icd10_map[x],drug_map[similar_words[i][0]],(n*100)/len(drug_map)])
                                                        #if len(top5) < 5:
                                                        #        top5.append([similar_words[i][1], drug_map[similar_words[i][0]]])
                                                        n = n + 1
                                        for i in range(len(d['drug'].unique().tolist())):
                                                if d['drug'].unique().tolist()[i] not in detected_drug:
                                                        drug_name = d['drug'].unique().tolist()[i]
                                                        if drug_name in drug_map:
                                                                drug_name = drug_map[drug_name]
                                                        data.append([txn, icd10_map[x], drug_name, len(drug_map)])
                                else:
                                        for i in range(len(d['drug'].unique().tolist())):
                                                drug_name = d['drug'].unique().tolist()[i]
                                                if drug_name in drug_map:
                                                        drug_name = drug_map[drug_name]
                                                icd10_name = x
                                                if x in icd10_map:
                                                        icd10_name = icd10_map[x]
                                                data.append([txn, icd10_name, drug_name, len(drug_map)])
                                        #txt = txt + str(icd10_map[dd]) + '\n'
                                        #txt = txt + '###All actual drugs###' + '\n'
                                        #txt = txt + str(d['drug_name'].unique().tolist()) + '\n'
                                        #txt = txt + '###Actual drug rank###' + '\n'
                                        #txt = txt + str(rank) + '\n'
                                        #txt = txt + '###Top 5 predicted drugs###' + '\n'
                                        #txt = txt + str(top5) + '\n\n'

                                #txt = txt + '#####################' + '\n'
                                #print(txt)
                                #with codecs.open(path + prefix+'validation.txt', 'a', encoding='utf8') as f:
                                #        f.write(txt)
                                #        f.close()
                result = pd.DataFrame(data, columns=['txn', 'icd10', 'drug_name', 'rank'])
                table = pd.pivot_table(result, values=['rank'], index=['icd10', 'drug_name'],
                                       aggfunc={'rank': [np.mean, np.min, np.max, np.count_nonzero]})
                if save:
                        result.to_csv(path + prefix+'validation_'+m+'.csv')
                        table.to_csv(path+prefix+'validation_pivot_'+m+'.csv')
                        print(result)
                        print(table)
                print(m)
                print(result['rank'].mean(axis=0))

        #df = pd.read_csv(path+'result.csv',index_col=0)
        #df.to_csv(path+'result.csv')

def save_projector():
        #word embedding projector
        df = pd.read_csv(path+'drug_name.csv')
        df['drug_name'] = df['drug_name'].str.strip()
        df['drug_name'] = df['drug_name'].apply(lambda x: re.sub('\n', '', str(x)))
        drug_map = dict(zip(df['drug'], df['drug_name']))
        model = gensim.models.Word2Vec.load(path+'model')
        icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
        icd10['cdesc'] = icd10['cdesc'].str.strip()
        icd10_map = dict(zip(icd10['code'], icd10['cdesc']))

        model = gensim.models.Word2Vec.load(path+'model')

        meta = []
        data = ''
        remove_file(path+'meta.tsv')
        remove_file(path+'data.tsv')
        for i in model.wv.vocab:
                found = False
                if i in drug_map:
                        meta.append(['drug',drug_map[i]])
                        found = True
                if i in icd10_map and not found:
                        meta.append(['icd10',icd10_map[i]])
                        found = True
                if found:
                        d = model.wv[i]
                        data = data + '\t'.join(str(x) for x in d) + '\n'

        meta_df = pd.DataFrame(meta, columns=['type','name'])
        meta_df.to_csv(path+'meta.tsv', sep = '\t', index=False)

        with codecs.open(path+'data.tsv', 'w', encoding='utf8') as f:
                f.write(data)
                f.close()

def save_rank():
        df = pd.read_csv(path+'drug_name.csv')
        df['drug_name'] = df['drug_name'].str.strip()
        df['drug_name'] = df['drug_name'].apply(lambda x: re.sub('\n', '', str(x)))
        drug_map = dict(zip(df['drug'], df['drug_name']))
        model = gensim.models.Word2Vec.load(path+'model')
        icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
        icd10['cdesc'] = icd10['cdesc'].str.strip()
        icd10_map = dict(zip(icd10['code'], icd10['cdesc']))

        model = gensim.models.Word2Vec.load(path+'model')
        data = []
        for i in model.wv.vocab:
                d = []
                found = False
                if i in drug_map:
                        d.append('drug')
                        d.append(i)
                        found = True
                if i in icd10_map and not found:
                        d.append('icd10')
                        d.append(i)
                        found = True
                if found:
                        similar_words = model.wv.most_similar(positive=[i], topn=5000)
                        n1 = 0
                        n2 = 0
                        drug_list = []
                        icd10_list = []
                        for j in range(len(similar_words)):
                                if similar_words[j][0] in drug_map:
                                        if n1 < 50:
                                                drug_list.append(similar_words[j][0])
                                                drug_list.append(similar_words[j][1])
                                        n1 = n1+1
                                if similar_words[j][0] in icd10_map:
                                        if n2 < 50:
                                                icd10_list.append(similar_words[j][0])
                                                icd10_list.append(similar_words[j][1])
                                        n2 = n2+1
                        while True:
                                if len(drug_list) < 100:
                                        drug_list.append(np.NaN)
                                else:
                                        break
                        while True:
                                if len(icd10_list) < 100:
                                        icd10_list.append(np.NaN)
                                else:
                                        break
                        d = d + drug_list + icd10_list
                        data.append(d)
                #break
        dru = []
        dx = []
        for i in range(50):
                dru.append('drug_'+str(i+1))
                dru.append('drug_s' + str(i + 1))
                dx.append('icd10_' + str(i + 1))
                dx.append('icd10_s' + str(i + 1))
        col = ['type','keyword']+dru+dx
        df = pd.DataFrame(data,columns=col)
        print(df)
        df.to_csv(path+'similarity.csv')

def save_rank_v(modelname):
        df = pd.read_csv(path+'drug_name.csv')
        df['drug_name'] = df['drug_name'].str.strip()
        df['drug_name'] = df['drug_name'].apply(lambda x: re.sub('\n', '', str(x)))
        drug_map = dict(zip(df['drug'], df['drug_name']))
        model = gensim.models.Word2Vec.load(path+modelname)
        icd10 = pd.read_csv(path + 'icd10.csv', index_col=0)
        icd10['cdesc'] = icd10['cdesc'].str.strip()
        icd10_map = dict(zip(icd10['code'], icd10['cdesc']))

        data = []
        for i in model.wv.vocab:
                type = ''
                keyword = ''
                found = False
                if i in drug_map:
                        type = 'drug'
                        keyword = i
                        found = True
                if i in icd10_map and not found:
                        type = 'icd10'
                        keyword = i
                        found = True
                if found and type != '' and keyword != '':
                        similar_words = model.wv.most_similar(positive=[i], topn=5000)
                        n1 = 0
                        n2 = 0
                        for j in range(len(similar_words)):
                                if similar_words[j][0] in drug_map:
                                        if n1 < 50:
                                                data.append([type,keyword,'drug',similar_words[j][0],similar_words[j][1]])
                                        n1 = n1+1
                                if similar_words[j][0] in icd10_map:
                                        if n2 < 50:
                                                data.append([type,keyword,'icd10',similar_words[j][0],similar_words[j][1]])
                                        n2 = n2+1

        col = ['type','keyword','type_s','keyword_s','similarity']
        df = pd.DataFrame(data,columns=col)
        df = df.sort_values(by=['type','keyword','type_s','keyword_s','similarity'], ascending=[True,True,True,True,False])
        df.index.name = 'id'
        print(df)
        df.to_csv(path+'similarity.csv')


#save_projector()
#save_rank()
#save_rank_v('model_58800')
#train_chain1('dru','model')
#train_chain1('idru','imodel')
#test_chain()
#test_chain(prefix='i')
#for i in range(100,61500,1000):
#        validate(['model_'+str(i)],n=10000,prefix='')
validate(['model_61000'],n=10000,prefix='',save=True)
#validate(prefix='i')

