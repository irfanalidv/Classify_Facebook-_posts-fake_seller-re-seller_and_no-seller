
# coding: utf-8

# In[2]:

#Reading data
import pandas as pd
from collections import Counter


# In[3]:

Fb_Data=pd.read_csv("FB_User_Classification.csv", delimiter="\t")


# In[119]:

Fb_Data.head()


# In[113]:

from langdetect import detect
from langdetect import detect_langs
lang=[]
[detect_langs(Fb_Data.description[2])]
        


# In[98]:

[detect_langs(Fb_Data.description[1])]


# In[96]:

detect_langs(Fb_Data.description[2])


# In[115]:

detect_langs(Fb_Data.description[122]) #Indo-European	Welsh	Cymraeg langauge


# In[116]:

Fb_Data['INDEX New'][122] #i think language has relation to find the fake people 


# In[82]:

Fb_Data['found_keywords_occurrences'].value_counts().head()


# In[28]:

Fb_Data['found_keywords'].value_counts().head()


# In[118]:

Fb_Data['pictures_url'].value_counts().head()


# In[77]:

Counter(Fb_Data['owner_type'])


# In[86]:

Fb_Data['nb_like'].value_counts().head() #i don't see the relevence like is it important | 
#actually can what if the fake user or seller is trying to buy the like can matter right


# In[88]:

Fb_Data['nb_share'].value_counts().head()


# In[42]:

Total_User =Fb_Data['INDEX New'].count()
Total_Type=pd.DataFrame(Fb_Data['INDEX New'].value_counts())
print(Total_User)
Total_Type


# In[63]:

Per_N0_Seller=round(100*(Total_Type['INDEX New'][0]/Total_User ),2)
print(Per_N0_Seller)
Per_Re_Seller=round(100*(Total_Type['INDEX New'][1]/Total_User ),2)
print(Per_Re_Seller)
Per_Fake_Seller=round(100*(Total_Type['INDEX New'][2]/Total_User ),2)
print(Per_Fake_Seller)


# In[ ]:



