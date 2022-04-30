import pandas as pd
import os

df_train = pd.DataFrame([], columns=['Filepath', 'Weight'])

df = pd.read_csv('./train/annotation.csv')
for i in df.iterrows():
    filename = i[1]['img'][:-4]
    weight = i[1]['weight']
    for file in os.listdir("./train"):
        if file.startswith(filename):
            # print(file)
            temp = pd.DataFrame([['./'+file, weight]], columns=['Filepath', 'Weight'])
            # print(temp)
            df_train = pd.concat([df_train, temp], ignore_index=True)
print(df_train)
df_train.to_csv('./train/_annotations.csv')
