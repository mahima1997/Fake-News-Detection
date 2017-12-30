import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


df  = pd.read_csv("/home/mahima/ML/Kaggle_assignments/fake_news/fake.csv")
#copy
df1=df

#subsetting
df = df[df['language'] == 'english']

#df[categorical_columns].apply(lambda x : print(x))
df.drop(['published','crawled','language','uuid','main_img_url','thread_title'], axis=1,inplace=True)

df.text.str.replace('[^\x00-\x7F]+', "")
df.title.str.replace('[^\x00-\x7F]+', "")

df.text.str.replace('(\\n)',"")
df.title.str.replace('(\\n)',"")

numerical_columns = list(df.select_dtypes(include=['int64']).columns)
categorical_columns = list(df.select_dtypes(include=['object']).columns)
text_columns=['title','text']
categorical_columns = [column for column in categorical_columns if column not in text_columns]

# fig, ax = plt.subplots()
# for n, col in enumerate(numerical_columns):
#     print(n,col)
#     ax.boxplot(df[col], positions=[n+1], notch=True)
#
# ax.set_xticks(range(6))
# ax.set_xticklabels(range(6))
# plt.show()

#Outliers removal
for i in numerical_columns:
    df = df[np.abs(df[i]-df[i].mean())<=(2*df[i].std())]
    outliers = df[np.abs(df[i]-df[i].mean())>=(2*df[i].std())]

############################################################
#converting all the numerical columns to categorical
###########################################################
for x in numerical_columns:
    try:
        df[x] = pd.qcut(df[x], 5, labels=["very low", "low", "medium","high","very high"])
    except Exception as e1:
        print(e1)
        try:
            df[x] = pd.cut(df[x], 5, labels=["very low", "low", "medium","high","very high"])
        except Exception as e2:
            print(e1,e2)

#as domain rank not in numerical columns as it is of series type
df['domain_rank'] = pd.qcut(df['domain_rank'], 5, labels=["very low", "low", "medium","high","very high"])

#Replacing lower frequency categorical columns to other
for x in categorical_columns:
    value = df[x].value_counts()
    if  np.percentile(df[x].value_counts().values,94.5) != np.percentile(df[x].value_counts().values,95.5):
        df[x] = df[x].replace({x: 'other' for x in value[value < np.percentile(df[x].value_counts().values,95)].index})
    else:
        print("see these columns i can't handle it",x)

#bcoz function can't handle site url
a=df.site_url.value_counts()
a=a[a<100].index
df.site_url=df.site_url.replace({x:'other' for x in a})

df2=df

df.T.drop_duplicates(inplace=True)

df3=df

cols_to_encode=categorical_columns+numerical_columns
df=pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode)
df=pd.get_dummies(df, columns=['domain_rank'], prefix=['domain_rank'])

#df.write.format("csv").save("/home/mahima/ML/Kaggle_assignments/fake_news/fake_clean.csv")

df.to_csv('/home/mahima/ML/Kaggle_assignments/fake_news/fake_clean.csv',index=False)

print()
