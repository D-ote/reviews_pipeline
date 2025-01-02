# %%
# %pip install google-api-python-client
# %pip install nltk
# %pip install contractions

# %%
import googleapiclient.discovery
import googleapiclient.errors

import pandas as pd
import psycopg2

# %% [markdown]
# #### Extract

# %%
import googleapiclient.discovery

# Set up API credentials
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyBhli20lTSkWKuiuxPjvvAc2TQn_Zn-YTY"

# Initialize the YouTube API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY
)

# Function to fetch all comments
def fetch_all_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Maximum allowed value
            pageToken=next_page_token
        )
        response = request.execute()

        # Extract comments from the response
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['publishedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])

        # Check if there's another page
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
        

    return comments

# Fetch comments for a video
video_id = "dtp6b76pMak"  
all_comments = fetch_all_comments(video_id)
print(f"Total Comments Fetched: {len(all_comments)}")


# %%
all_comments

# %%
df = pd.DataFrame(all_comments, columns=['dated', 'likes', 'text'])

# %%
df.info()

# %% [markdown]
# #### Transform

# %%
# Cleaning libraries

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions

# %%
# Remove HTML tags
def remove_html_tags(text):
    html_tag_pattern = re.compile(r'<a href=".*?">.*?</a>')  
    return html_tag_pattern.sub('', text)

# Apply the function to the DataFrame
df['text'] = df['text'].apply(remove_html_tags)
df.head()

# %%
# convert the date column to datetime format
df['dated'] = pd.to_datetime(df['dated'])

# change the datetime format
df['date'] = df['dated'].dt.strftime('%Y-%m-%d')

# %%
# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords list
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    #fix contracted words
    text = contractions.fix(text)
    
    # Remove mentions, hashtags, URLs
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text, preserve_line=True)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to a single string
    clean_text = ' '.join(tokens)
    
    return clean_text

# Apply the cleaning function to your text column
df['cleaned_text'] = df['text'].apply(clean_text)

# View the cleaned data

df[['text', 'cleaned_text']].head()

# %%
df.head()

# %% [markdown]
# #### Load

# %%
conn = psycopg2.connect(
    dbname="api_data", user="postgres", password="Wueseter1!", host="localhost", port="5432"
)
cur = conn.cursor()

for _, row in df.iterrows():
    cur.execute(
        "INSERT INTO vision_pro_reviews (date, review) VALUES (%s, %s)",
        (row['date'], row['cleaned_text'])
    )
print('successfully loaded')
conn.commit()
cur.close()
conn.close()

# %%
# df = pd.read_sql_query("SELECT * FROM yt_feedback", conn)

# %% [markdown]
# #### Automate

# %%
# crontab -e

# 0 0 * * * /Users/dooterior/Desktop/projects/pipeline/pipeline/bin/python /Users/dooterior/Desktop/projects/pipeline/ytcomments.py

#:wq


