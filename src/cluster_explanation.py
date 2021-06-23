#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import e
from pymining import itemmining,assocrules
import re
import operator

df=pd.read_csv('travel-case-200-bd.csv')

''' Adding cluster label '''
cluster_ids=[]
case_ids=df['traceid'].unique()
for case_id in case_ids:
    cluster_id=np.random.choice(['cluster-1','cluster-2','cluster-3'],1,True,[0.6,0.3,0.1])[0]
    size_=df[df['traceid']==case_id].shape[0]
    cluster_ids.extend([cluster_id for i in range(size_)])
df['cluster']=cluster_ids


def data_conversion(cluster_df):
    cluster_df=cluster_df.fillna("NA")
    cols=list(cluster_df.columns)[:-1]
    for col in cols:
        if type(list(cluster_df[col])[0])==str:
            pass
        else:
            cluster_df[col]=pd.qcut(cluster_df[col], q=4,duplicates='drop') 
    return cluster_df

cluster_df=df[df.columns[9:]]
processed_df=data_conversion(cluster_df)

''' 
##########################
item set based method 
##########################
'''
def get_item_sets(df):
    item_set=[]
    df_cols=list(df.columns)[:-1]
    for i,rec in df.iterrows():
        temp_rec=[]
        for col in df_cols:
            temp_rec.append(str(col)+"="+str(rec[col]))
        item_set.append(tuple(temp_rec))
    return item_set

def get_rules(item_set,top_n=5):
    transactions = tuple(item_set)
    relim_input = itemmining.get_relim_input(transactions)
    item_rules= itemmining.relim(relim_input, min_support=int(0.2*len(item_set)))
    rule_list=[]
    for ir in item_rules:
        rule_list.append([tuple(ir),item_rules[ir]])
    sorted_rule_list=sorted(rule_list,key=lambda x: x[1],reverse=True)
    return sorted_rule_list[:top_n]

def cluster_explanation_item_set(df):
    df = processed_df  
    cluster_labels= list(df['cluster'].unique())
    explanations={}
    for cl in cluster_labels:
        item_set=get_item_sets(df[df['cluster']==cl])
        rules = get_rules(item_set)
        explanations[cl]=rules
    return explanations
        
def clsuter_explanation_assoc_rules(df):
    df = processed_df  
    cluster_labels= list(df['cluster'].unique())
    explanations={}
    for cl in cluster_labels:
        #cl = cluster_labels[0]
        item_sets=get_item_sets(df[df['cluster']==cl])
        cut_off=0.5
        min_support= int(cut_off*len(item_sets))
        transactions = tuple(item_sets)
        relim_input = itemmining.get_relim_input(transactions)
        item_sets = itemmining.relim(relim_input, min_support=min_support)
        rules = assocrules.mine_assoc_rules(item_sets, min_support=min_support, min_confidence=0.95)
        records_to_write=[]
        for rl in rules:
            temp=[]
            for rle in rl:
                if type(rle)==frozenset:
                    temp.append(list(rle))
                else:
                    temp.append(rle)
            records_to_write.append(temp)
        explanations[cl]=records_to_write[-5:]
        
                

''' 
#########################
Entropy based method 
#########################
'''
def column_entropy(column, base=None):
  vc = pd.Series(column).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc+0.0000001)/np.log(base+0.0000001)).sum()

column_entropy([0.66,0.33],base=2)
column_entropy([0.63,0.36],base=2)

def split_df(df,col):
    return [pd.DataFrame(y) for x, y in df.groupby(col, as_index=False)]

def get_column_with_min_entropy(df,cols):
    #df,cols=required_cluster_df,col_names
    min_col=None 
    max_entropy=10000000
    for col in cols:
        entropy=column_entropy(df[col])
        if entropy<max_entropy:
            min_col=col
            max_entropy=entropy
        else:
            pass
    return min_col


def get_column_with_max_entropy(df,cols):
    #df,cols=processed_df,processed_df.columns[:-1]
    max_col=None 
    min_entropy=0
    for col in cols:
        entropy=column_entropy(df[col])
        if entropy>min_entropy:
            max_col=col
            min_entropy=entropy
        else:
            pass
    return max_col


def min_entropy_rule(df,threshold=200,rule=''):
    col_names=list(df.columns)
    col_with_min_entropy = get_column_with_min_entropy(df,col_names)
    df_list=split_df(df,col_with_min_entropy)
    for df_i in df_list:
        if df_i.shape[0]>threshold:
            rule = rule+"+"+str(col_with_min_entropy)+"="+str(list(df_i[col_with_min_entropy].unique())[0])+"{"+str(df_i.shape[0])+"}"
            del df_i[col_with_min_entropy]
            min_entropy_rule(df_i,rule=rule)
            temp_rules.append(rule)

def brace_split(ta):
    var, value = ta.split("{")
    return [var,int(re.sub('}','',value))]

def rules_hierarchy(temp_rules):
    rule_set={}
    for tr in temp_rules:
        temp_arr=[brace_split(t) for t in tr.split("+") if t!='']
        for ta in temp_arr:
            if ta[0] in rule_set:
                pass
            else :
                rule_set[ta[0]]=ta[1] 
    return sorted(rule_set.items(),key=operator.itemgetter(1),reverse=True)
        

def cluster_entropy_rules():
    cluster_rules=[]
    min_records_fraction=0.10
    for cluster_label in processed_df['cluster'].unique():
        #cluster_label='cluster-1'
        #print(cluster_label)
        temp_rules=[]
        temp_df=processed_df[processed_df['cluster']==cluster_label]
        del temp_df['cluster']
        cluster_size=temp_df.shape[0]
        min_recs=int(cluster_size*min_records_fraction)
        min_entropy_rule(temp_df,threshold=min_recs)
        cluster_rules.append({'cluster_label':cluster_label,'cluster_size':cluster_size,'rules_hierarchy':rules_hierarchy(temp_rules)})
    print(cluster_rules)
        
    
    
       
        


