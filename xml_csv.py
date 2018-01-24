import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np

tree = ET.parse("MVI_20011.xml")
root = tree.getroot()
box = {}
attributes={}
ans = []


i=0
j=0

frm=[]
for child in root:
	for x in child:
		for y in x:
			for z in y:
				if z.tag=='box':
					box[i]=dict(z.attrib.items()+child.attrib.items())
					i=i+1
				if z.tag=='attribute':
					attributes[j]=z.attrib
					j=j+1
	
df = pd.DataFrame.from_dict(box,orient='index')
df1 = pd.DataFrame.from_dict(attributes,orient='index')
df2 = pd.concat([df,df1],axis=1)
df2.vehicle_type.replace(to_replace=dict(car=0,van=1,bus=2,others=3), inplace=True) #Vehicle type mapped to integer value
d=df2.groupby('num')['num'].apply(list)
for q in range(0,len(d)):
	ans.append(d[q][0])
se=pd.Series(ans)
se=pd.to_numeric(se)
# d0=df2.groupby('num')['top'].apply(list)
d1=df2.groupby('num')['top'].apply(list)
d2=df2.groupby('num')['height'].apply(list)
d3=df2.groupby('num')['width'].apply(list)
d4=df2.groupby('num')['left'].apply(list)
d5=df2.groupby('num')['speed'].apply(list)
d6=df2.groupby('num')['orientation'].apply(list)
d7=df2.groupby('num')['trajectory_length'].apply(list)
fin = pd.concat([d1,d2,d3,d4,d5,d6,d7],axis=1)
# fin.sort_values('frame_number',ascending=True)
fin['frame_number']=se.values
# fin.set_index('frame_number')
f=pd.DataFrame(fin)

f['frame_number'].astype(str).astype(float)
f.sort_values(by='frame_number', ascending=True)
# print f['frame_number'].dtypes
f.to_csv("seq1.csv")
# print fin[['num']]
df3 = pd.read_csv("MVI_20011.csv")
