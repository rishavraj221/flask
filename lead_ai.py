import logging
from flask import Flask, Response, stream_with_context, request, jsonify
from services.scraper.reddit import RedditScraper
from data_structure.Collection import Collection
from data_structure.Source import Source
from services.sentiment_analysis import EmotionAnalyzer
import json
from openai import OpenAI
from settings import OPENAI_KEY, OPENAI_ORG, OPENAI_PROJECT, MODEL_NAME
import uuid
from pydantic import BaseModel
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

llm_client = OpenAI(api_key=OPENAI_KEY,
                    project=OPENAI_PROJECT, organization=OPENAI_ORG)

scraper = RedditScraper(user_agent="SyncLoom")
source = Source()
source_base_url = "https://www.reddit.com"
source.set_source(name="Reddit", url=source_base_url)

emotion_analyser = EmotionAnalyzer()
database = []
local_data = {}


class RedditResponseFormat(BaseModel):
    subreddits: list[str]
    keywords: list[str]


@app.route('/', methods=["GET"])
def hello():

    return jsonify({"status": "success", "message": "hello from Lead-AI development api!"})


@app.route('/save-data', methods=['POST'])
def save_data():

    data = request.get_json()

    unique_id = uuid.uuid4()

    local_data[str(unique_id)] = data

    return jsonify({"status": "success", "data_storage_id": unique_id})


@app.route('/scrape-sse', methods=['GET'])
def scrape_reddit():

    local_data_storage_id = request.args.get('storage-id')
    business_data = local_data.get(local_data_storage_id)

    if not business_data:
        return jsonify({"error": "Business data not found!"}), 400

    def generate(business_data):

        yield f"""data: {json.dumps({"type": "text", "data": "Analysing business ..."})}\n\n"""
        completion = llm_client.beta.chat.completions.parse(
            model='gpt-4o',
            messages=[
                {
                    'role': 'system',
                    'content': f"""
You are an expert in digital marketing and online community engagement. 
Your task is to help users identify potential clients by suggesting relevant subreddits and keywords based on the details of their business and target audience.

When a user provides details, follow these steps:

1. Understand the Business Context:
    - Identify the problem the business solves.
    - Note the industry or niche (e.g., fitness, tech, mental health, finance, etc.).
    - Determine the audience demographics (e.g., age, interests, profession, location, pain points, etc.).

2. Generate Subreddits:
    - Suggest 3-5 subreddits where the target audience might actively discuss or seek solutions to the problems the business addresses.
    - Prioritize subreddits that are highly active, niche-relevant, and likely to have members who need the business's solutions.

3. Suggest Keywords:
    - Provide 5-6 keywords or phrases that are commonly used in discussions about the problem the business solves.
    - Keywords should align with audience search intent, focusing on questions, challenges, or solutions related to the business niche.

Key Considerations:
- Be specific to the user’s business. For example, if the business is a mental health app for young adults, suggest subreddits like 'r/Anxiety' or 'r/MentalHealth' and keywords like 'coping with anxiety' or 'stress management for students.'
- Ensure the suggestions are current and actionable, targeting the platforms and keywords where meaningful engagement is likely to occur.

Return in the following format - 
subreddits: list[str]
keywords: list[str]
"""
                },
                {
                    'role': 'user',
                    'content': f"""
**Business Name:**
{business_data.get('business_name')}

**Business Description:**
{business_data.get('business_description')}

**Target Audience:**
{business_data.get('target_audience')}
"""
                }
            ],
            temperature=0.2,
            response_format=RedditResponseFormat
        )
        response_content = completion.choices[0].message.parsed

        subreddits = response_content.subreddits
        keywords = response_content.keywords

        yield f"""data: {json.dumps({"type": "text", "data": f"Will fetch data from subreddits: {', '.join(subreddits)} with keywords: {', '.join(keywords)}"})}\n\n"""
        logger.info(f"Subreddits: {subreddits}")
        logger.info(f"Keywords: {keywords}")

        count = 1
        for subreddit in subreddits:
            yield f"""data: {json.dumps({"type": "text", "data": f"Processing subreddit '{subreddit}' ({count} / {len(subreddits)}) ..."})}\n\n"""
            logger.info(f"Processing subreddit {count} / {len(subreddits)}")
            count += 1

            keyword_count = 1
            for keyword in keywords:
                yield f"""data: {json.dumps({"type": "text", "data": f"Processing keyword '{keyword}' ({keyword_count} / {len(keywords)}) in subreddit '{subreddit}' ..."})}\n\n"""
                logger.info(
                    f"Processing keyword {keyword_count} / {len(keywords)} in subreddit {subreddit}")
                keyword_count += 1

                try:
                    results = scraper.search_reddit_keyword(
                        subreddit.replace('r/', ''), keyword)
                except Exception as exc:
                    logger.error(f"Fetching posts error: {str(exc)}")

                yield f"""data: {json.dumps({"type": "text", "data": f"Found {len(results)} results for keyword '{keyword}' in subreddit '{subreddit}."})}\n\n"""
                logger.info(f"Results size: {len(results)}")

                for result in results:
                    author_name = result.get(
                        'author', {}).get('name', 'unknown')

                    collection = Collection(source)
                    collection.set_collection(
                        content=result.get('title'),
                        url=result.get('url'),
                        author_name=author_name,
                        author_url=f"{source_base_url}/user/{author_name}",
                    )

                    yield f"""data: {json.dumps({"type": "text", "data": f"Analyzing sentiments for post '{result.get('title')} ..."})}\n\n"""
                    logger.info(
                        f"Setting sentiments for post '{result.get('title')}'")
                    sentiments = emotion_analyser.analyze_emotion(
                        result.get('title'))
                    collection.set_sentiment(sentiments)

                    yield f"""data: {json.dumps({"type": "json_data", "data": collection.get_whole_collection()})}\n\n"""
                    # database.append(collection)

                    try:
                        comments = scraper.fetch_comments(
                            result.get('comments_url'))
                    except Exception as exc:
                        logger.error(f"Fetching comment error : {str(exc)}")

                    if len(comments) > 0:
                        yield f"""data: {json.dumps({"type": "text", "data": f"Processing {len(comments)} comments for post '{result.get('title')} ..."})}\n\n"""
                    else:
                        yield f"""data: {json.dumps({"type": "text", "data": f"No comments found for post '{result.get('title')}."})}\n\n"""
                    logger.info(
                        f"Processing {len(comments)} comments for post '{result.get('title')}'")

                    for comment in comments:
                        comment_author_name = comment.get(
                            'author', {}).get('name', 'unknown')
                        collection2 = Collection(source)
                        collection2.set_collection(
                            content=comment.get('comment'),
                            url=f"{source_base_url}/{comment.get('permalink')}",
                            author_name=comment_author_name,
                            author_url=f"{source_base_url}/user/{comment_author_name}",
                        )
                        sentiments = emotion_analyser.analyze_emotion(
                            comment.get('comment'))
                        collection2.set_sentiment(sentiments)

                        yield f"""data: {json.dumps({"type": "json_data", "data": collection2.get_whole_collection()})}\n\n"""
                        # database.append(collection2)

        yield f"""data: {json.dumps({"type": "done", "data": "Scraping Complete."})}\n\n"""
        logger.info("Scraping complete.")

    return Response(stream_with_context(generate(business_data)), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5465)
