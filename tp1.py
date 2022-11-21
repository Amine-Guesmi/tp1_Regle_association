#import Bib
from ast import Index
import os
import pandas
import numpy
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from ast import Index

def is_inclus(x,items):
 return items.issubset(x)

D = pandas.read_table("market_basket.txt",delimiter="\t",header=0) #Read Dataframe

print(D.head(10)) #show 10 first lines
print(D.shape) #show dimension of this df

TC = pandas.crosstab(D.ID,D.Product) 
print(TC.iloc[:30,:3])

freq_itemsets = apriori(TC,min_support=0.025,max_len=4,use_colnames=True)
type(freq_itemsets)

print(freq_itemsets.head(15))

id = numpy.where(freq_itemsets.itemsets.apply(is_inclus,items={'Aspirin'}))
print(freq_itemsets.loc[id])

print(freq_itemsets[freq_itemsets['itemsets'].eq('Aspirin')])
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin','Eggs'})])

regles = association_rules(freq_itemsets,metric="confidence",min_threshold=0.75)

print(type(regles))
print(regles.shape)
print(regles.columns)
print(regles.iloc[:5,:])

myRegles = regles.loc[:,['antecedents','consequents','lift']]

print(myRegles[myRegles['lift'].ge(7.0)])
print(myRegles[myRegles['consequents'].eq({'2pct_Milk'})])