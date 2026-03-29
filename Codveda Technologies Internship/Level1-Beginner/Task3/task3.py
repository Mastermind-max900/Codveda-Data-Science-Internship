import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading cleaned data from task 2
df = pd.read_csv('cleaned_sentiment_dataset.csv')

#Comput summary statistics(objective 1)
stats = df[['Likes', 'Retweets']].describe()
print("Summary Statistics:\n", stats)

#Visualize distributions(objective 2)
plt.figure(figsize=(10, 6))
sns.histplot(df['Likes'], kde=True, color='blue')
plt.title('Distribution of Likes')
plt.savefig('likes_distribution.png') #Saves the chart as an image file
plt.show()

#Correlation Matrix(objective 3)
plt.figure(figsize=(8, 6))
correlation_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.show()