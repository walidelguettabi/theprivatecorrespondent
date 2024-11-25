from datetime import date, timedelta
from src.newsutils_lambda import *

openai_api_key = "API-KEY"
client = OpenAI(api_key=openai_api_key)

def get_full_prompt(user_prompt):
    full_prompt = f"""  Your summary should invoke or partially quote elements of the news list.
                        Start the summary with a lighthearted observation /
                        about the day's events, or give general fun fact, the goal is to create a sense of connection with the reader, /
                        making them feel as though they're being spoken to directly, almost like a friendly conversation.
                        Focus and prioritize news that appear to be important, impactful, and appear multiple times on the list.
                        The user is interested in: {user_prompt}.
                        Your summary must be clear, concise and easy to understand.
                        Your text must not be a list of bullet points.
                        """
    return full_prompt


EMAIL_ADDRESS = "pricorr6@gmail.com"
EMAIL_PASSWORD = "EMAIL_PASSWORD"

with open('data/rss_sources.json', 'r') as json_file:
    rss_urls = json.load(json_file)

with open('data/users.json', 'r') as json_file:
    users = json.load(json_file)


def lambda_handler(event, context):
    print("---- DOWNLOADING NEWS ----")
    rss_df = get_news_list(rss_urls)
    print("---- NEWS SUCCESSFULLY DOWNLOADED ----")
    dt = (date.today()- timedelta(days=1)).strftime('%Y-%m-%d')
    rss_df = rss_df[rss_df['pubDate']>=dt]
    news_list = rss_df['news'].values
    print("---- GET NEWS EMBEDDINGS")
    np_vectors = news_embeddings(client, news_list)
    print("---- EMBEDDING SUCCEEDED ----")

    for user in users:
        user_prompt = user['prompt']
        keywords = generate_keywords(client, user_prompt)
        keywords_embedding_vector = keywords_embedding(client, keywords)
        distances = get_distances(np_vectors, keywords_embedding_vector, news_list)
        filtered_news_list = list(distances.sort_values(by='dist', ascending=False).iloc[0:100]['story'].values)
        summary = f"Good Morning {user['username']} ! \n\n" + generate_summary(client, get_full_prompt(user_prompt), filtered_news_list)
        title = make_title(client, summary)
        send_email(EMAIL_ADDRESS, EMAIL_PASSWORD, user['email'],title, summary)
        send_email(EMAIL_ADDRESS, EMAIL_PASSWORD, 'w.elguettabi@gmail.com',title, summary)
        print("Ending the script")


    return {
        'statusCode': 200,
        'body': json.dumps('Done')
    }