import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('3) Sentiment dataset.csv')

df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

df.dropna(inplace=True)

mean_likes = df['Likes'].mean()
std_likes = df['Likes'].std()
df = df[(df['Likes'] <= mean_likes + 3 * std_likes)]
le = LabelEncoder()
df['Sentiment_Encoded'] = le.fit_transform(df['Sentiment'])
df['Platform_Encoded'] = le.fit_transform(df['Platform'])

df.to_csv('cleaned_sentiment_dataset.csv', index=False)
print("Task 2: Data Cleaning Complete!")