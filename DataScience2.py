#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd

# dff = df.to_csv("QueryResults.csv", header=False, index=False)

colnames=['DATE', 'TAG', 'POSTS']

df = pd.read_csv("QueryResults.csv", names=colnames, header=None)

print(df)


# In[60]:


df.head()


# In[7]:


df.tail()


# In[8]:


print(f"The number of rows is {df.shape[0]}, The number of columns is {df.shape[1]}")


# In[9]:


df1 = pd.DataFrame(df)
result = df1.count(axis=1)
print(result)


# In[10]:


df.groupby("TAG").sum()


# In[12]:


df.groupby("TAG").count()


# In[13]:


df["DATE"][1]


# In[14]:


df.DATE[1]


# In[15]:


type(df["DATE"][1][0])


# In[62]:


# df1 = pd.DataFrame(df.DATE[1])
print(df1["DATE"])


# In[64]:


dd = pd.to_datetime(df1["DATE"])
print(dd)


# In[54]:


test_df = pd.DataFrame({"Age": ["young", "young","young", "young", "old", "old","old", "old"],
                      "Actor": ["Jack", "Arnorld", "Keanu", "Sylvester", "Jack", "Arnorld", "Keanu", "Sylvester"],
                        "Power": [100, 80, 25, 50, 99, 75, 5, 99]
                       })

df = pd.DataFrame({'foo': ['young', 'young', 'young', 'young', 'old', 'old', 'old','old'],
                   'bar': ["Jack", "Arnorld", "Keanu", "Sylvester", "Jack", "Arnorld", "Keanu", "Sylvester"],
                   'baz': [10, 20, 83, 94, 5, 6, 89, 87],
                  })
print(test_df)


# In[55]:


# test_df.pivot(index="Age", columns="Actor", values="Power")
test_df.pivot(index="Age", columns="Actor", values="Power")
# df.pivot(index='foo', columns='bar', values='baz')


# In[56]:


df.pivot(index='foo', columns='bar', values='baz')


# In[61]:


df.head()


# In[65]:


dd.head()


# In[68]:


df.head()
#pd.to_datetime(df["DATE"])


# In[69]:


df.DATE = pd.to_datetime(df.DATE)


# In[70]:


df.head()


# In[73]:


reshaped_df = df.pivot(index="DATE", columns="TAG", values="POSTS")
print(reshaped_df)


# In[75]:


reshaped_df.shape


# In[76]:


reshaped_df.head()


# In[77]:


reshaped_df.tail()


# In[78]:


for col in reshaped_df.columns:
    print(col)


# In[84]:


print(len(reshaped_df['c#']))


# In[85]:


print(len(reshaped_df['c']))


# In[86]:


print(len(reshaped_df['swift']))


# In[87]:


reshaped_df.count()


# In[88]:


reshaped_df = reshaped_df.fillna(0)


# In[89]:


reshaped_df.head()


# In[90]:


reshaped_df.index


# In[110]:


import matplotlib.pyplot as plt
roll_df = reshaped_df.rolling(window=20).mean()

plt.figure(figsize=(16, 10))
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.ylim(0, 35000)

for column in roll_df.columns:
    plt.plot(roll_df.index, roll_df[column], linewidth=3,
            label=roll_df[column].name)
    
plt.legend(fontsize=16)


# so what the fuck

# In[111]:


# so wat the fuck


# # so what the fuck

# <img src="bricks.png">

# ![Python logo](http://localhost:8888/files/lego_sets.png)

# ![Python logo][http://localhost:8888/files/rebrickable_schema.png]

# ![Python logo](http://localhost:8888/files/rebrickable_schema.png)

# <h1> This is header One </h1>

# In[113]:


import pandas as pd

# dff = df.to_csv("QueryResults.csv", header=False, index=False)

# colnames=['DATE', 'TAG', 'POSTS']

dfc = pd.read_csv("colors.csv")

print(dfc)


# In[114]:


dfc = pd.DataFrame(dfc)


# In[115]:


dfc.nunique()


# In[117]:


dfc.groupby("is_trans").count()


# ** i**

# <h1><b> Understanding LEGO Themes vs. LEGO Sets </b></h1>
# 
# <h2>
#     Walk into a LEGO store and you will see their products organised by theme. Their themes include Star Wars, Batman,
#     Harry Potter, and many more.
# </h2>
# 
# ![Python logo](http://localhost:8888/files/lego_themes.png)
# 
# <h3> 
# A lego is a particular box of LEGO or product. Therefore, a single theme typically has have many
# diffrents sets
# </h3>
# 
# ![Python logo](http://localhost:8888/files/lego_sets.png)
# 

# In[120]:


dfs = pd.read_csv("sets.csv")


# In[128]:


dfs


# In[121]:


dfs.head()


# In[125]:


print(f"The year is {dfs['year'][0]} and name of the product is {dfs['name'][0]}")


# In[126]:


dfs['num_parts'][0]


# In[130]:


dfs['num_parts'].max()


# In[131]:


dfs.sort_values('year').head()


# In[132]:


dfs[dfs['year']==1949]


# In[134]:


dfs.sort_values('num_parts', ascending=False).head()


# In[151]:


import matplotlib.pyplot as plt

dfs_year = dfs.groupby('year').count()


# In[152]:


dfs_year


# In[160]:


plt.plot(dfs_year.set_num)


# In[177]:


import numpy as np
plt.plot(np.linspace(0, 800, 69), dfs_year.set_num[:-2]) #, 1995)


# In[167]:


len(dfs_year.set_num[:-2])


# In[194]:


dfs


# In[179]:


themes_by_year = dfs.groupby('year').agg({'theme_id': pd.Series.nunique})


# In[184]:


themes_by_year


# In[192]:


plt.plot(themes_by_year.index[:-2], themes_by_year.theme_id[:-2])


# In[203]:


themes_by_year.theme_id[:-2]


# In[198]:


ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(themes_by_year.index[:-2], themes_by_year.theme_id[:-2], color='r')
ax2.plot(dfs_year.index[:-2], dfs_year.set_num[:-2], color='b')

ax1.set_xlabel('year')
ax1.set_ylabel('Number of themes', color='red')
ax2.set_ylabel('Number of set', color='blue')


# In[199]:


parts_per_set = dfs.groupby('year').agg({'num_parts':pd.Series.mean})


# In[200]:


parts_per_set.head()


# In[201]:


parts_per_set.tail()


# In[202]:


plt.scatter(parts_per_set.index[:-2], parts_per_set.num_parts[:-2])


# In[205]:


set_theme_counts = dfs["theme_id"].value_counts()


# In[206]:


set_theme_counts


# ![Python logo](http://localhost:8888/files/rebrickable_schema.png)

# In[209]:


dft = pd.read_csv("themes.csv")


# In[210]:


dft


# In[213]:


dft[dft.name == 'Star Wars']


# In[214]:


dfs[dfs.theme_id == 18]


# In[215]:


dfs


# In[219]:


set_theme_count = pd.DataFrame({'id':set_theme_counts.index, 'set_count':set_theme_counts.values})


# In[220]:


set_theme_count


# In[221]:


merged_df = pd.merge(set_theme_count, dft, on='id')

merged_df[:3]


# In[224]:


plt.bar(merged_df.name[:10], merged_df.set_count[:10])


# In[225]:


plt.figure(figsize=(14, 8))
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.ylabel('Nr of Sets', fontsize=14)
plt.xlabel('Theme Name', fontsize=14)

plt.bar(merged_df.name[:10], merged_df.set_count[:10])


# In[ ]:




