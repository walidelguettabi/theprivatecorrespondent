import requests
import pandas as pd
from openai import OpenAI
import numpy as np
import xml.etree.ElementTree as ET
import json
#from tqdm import tqdm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import *

def get_news_from_rss(url):
    # Sample RSS feed XML (truncated for simplicity in this example)
    rss_feed = requests.get(url).content
    # Parse the XML
    root = ET.fromstring(rss_feed)
    # Extract relevant data and format it as JSON
    rss_json = {"items": []}
    for item in root.findall("./channel/item"):
        title = item.find("title").text.strip()
        description = item.find("description").text.strip()
        pubDate = item.find("pubDate").text.strip()
        rss_json["items"].append({"title": title, "description": description, "pubDate": pubDate})

    # Convert to JSON string
    #rss_json_str = json.dumps(rss_json, indent=4, ensure_ascii=False)
    return rss_json['items']

def get_news_from_rss_2(url):

    # Fetch and parse the RSS feed
    rss_feed = requests.get(url).content
    root = ET.fromstring(rss_feed)

    # Convert RSS feed to JSON
    rss_json = {"items": []}
    # Iterate through each item in the RSS feed
    for item in root.findall("./channel/item"):
        title = item.find("title")
        description = item.find("description")
        pubDate = item.find("pubDate")
        
        # Use conditional expressions to handle any missing fields
        item_data = {
            "title": title.text if title is not None else "No title",
            "description": description.text if description is not None else "No description",
            "pubDate": pubDate.text if pubDate is not None else "No publication date"
        }
        
        rss_json["items"].append(item_data)

    return rss_json['items']


def get_news_list(rss_urls_list):
    rss_news_list = []
    for url in rss_urls_list:
        rss_news_list = rss_news_list + get_news_from_rss_2(url)

    rss_df = pd.DataFrame(rss_news_list)
    rss_df['pubDate'] = pd.to_datetime(rss_df['pubDate'], format='mixed', yearfirst=True)
    rss_df['pubDate'] = rss_df['pubDate'].apply(lambda x: str(x).split(' ')[0])
    rss_df['news'] = 'Title: '+rss_df['title'].astype(str) + '. Description: '+rss_df['description'].astype(str)


    return rss_df

def news_embeddings(client, news_list):
    vectors = []
    for story in news_list:
        embedding_response = client.embeddings.create(
                input=story,
                model="text-embedding-3-small"
            )
        vectors.append(embedding_response.data[0].embedding)
    return np.array(vectors)

def generate_keywords(client, user_prompt):

    messages = [
        {
            "role": "system",
            "content": "You're a journalist.'"
        },

        {
            "role": "user", 
            "content":  f"""\
                        Give 100 words related to this sentence: {user_prompt}.
                        Write them in one line.
                        """ 
        }
    ]


    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 1.2,
        messages= messages
    )

    return completion.choices[0].message.content

def make_title(client, summary):

    messages = [
        {
            "role": "system",
            "content": "You're a journalist.'"
        },

        {
            "role": "user", 
            "content":  f"""\
                        Give a title to this: {summary}.
                        Do not put it between quotes or brackets.
                        Be creative. You don't have to use all the elements
                        from the text to come up with a title.
                        """ 
        }
    ]


    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 1.2,
        messages= messages
    )

    return completion.choices[0].message.content

def keywords_embedding(client, keywords):
    prompt_embedding = client.embeddings.create(
            input=keywords,
            model="text-embedding-3-small"
        ).data[0].embedding
    return np.array(prompt_embedding)

def get_distances(np_vectors, keywords_embedding, news_list):
    distances = {'story':[], 'dist':[]}
    for i in range(0,np_vectors.shape[0]):
        #d = np.linalg.norm(np.array(prompt_embedding-np_vectors[i]))
        d = np.dot(keywords_embedding,np_vectors[i])
        distances['story'].append(news_list[i])
        distances['dist'].append(d)
    distances = pd.DataFrame(distances)
    return distances
    



def generate_summary(client, user_prompt, news_list):

    messages = [
        {
            "role": "system",
            "content": "Your job is to write news summaries."
        },

        {
            "role": "user", 
            "content":  f"""\
                        Here is a list of the main news from today: {str(news_list)}.
                        """ + user_prompt
        }
    ]


    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 1.2,
        messages= messages
    )

    return completion.choices[0].message.content

# Function to send email
def send_email(EMAIL_ADDRESS, EMAIL_PASSWORD, recipient, subject, content):
    # Set up email details
    msg = MIMEMultipart()
    msg['From'] = "The Private Correspondent"
    msg['To'] = recipient
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(content, 'plain'))

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())
        print(f"Email sent to {recipient}")
    except Exception as e:
        print(f"Error sending email to {recipient}: {e}")