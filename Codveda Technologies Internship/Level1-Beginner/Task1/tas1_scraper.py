import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

url = "https://quotes.toscrape.com"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

scraped_data = []
quotes = soup.find_all('div', class_='quote')

for q in quotes:
    text = q.find('span', class_='text').get_text()
    author = q.find('small', class_='author').get_text()
    
    tags = [tag.get_text() for tag in q.find_all('a', class_='tag')]
    hashtags = " ".join([f"#{t}" for t in tags])
    scraped_data.append({
        'Text': text,
        'Author': author,
        'Hashtags': hashtags,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Platform": "WebScraper"
    })
df = pd.DataFrame(scraped_data)
df.to_csv(r'C:\Users\ZBook\Desktop\my_sentiment_scrape.csv', index=False)
print("Scraping completed and data saved to my_sentiment_scrape.csv")