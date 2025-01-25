import praw
import csv
import logging
from datetime import datetime
from settings import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(
        "reddit_scraper.log"), logging.StreamHandler()],
)


class RedditScraper:
    def __init__(self, user_agent):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=user_agent,
        )

    def get_author_info(self, author_name):

        author = self.reddit.redditor(author_name)

        return {
            "name": getattr(author, "name", "Unknown"),
            "created_utc": getattr(author, "created_utc", None),
            "account_info": {
                "is_moderator": getattr(author, "is_mod", False),
                "is_employee": getattr(author, "is_employee", False),
                "is_gold": getattr(author, "is_gold", False),
                "has_verified_email": getattr(author, "has_verified_email", False)
            },
            "activity": {
                "link_karma": getattr(author, "link_karma", 0),
                "comment_karma": getattr(author, "comment_karma", 0),
                "total_karma": getattr(author, "total_karma", 0)
            }
        }

    def search_reddit_keyword(self, subreddit, keyword):
        logging.info(
            f"Searching subreddit: {subreddit} and keyword: {keyword}")

        results = []

        for post in self.reddit.subreddit(subreddit).search(keyword):
            results.append({
                "title": post.title,
                "url": post.url,
                "author": self.get_author_info(post.author.name),
                "created_utc": post.created_utc,
                "comments_url": post.permalink,
                "num_comments": post.num_comments
            })

        return results

    def search_reddit(self, subreddit, keywords, limit=50):
        """Search posts in a subreddit based on keywords."""
        logging.info(f"Searching subreddit: {subreddit}")
        results = []
        for keyword in keywords:
            try:
                logging.info(f"Searching for keyword: {keyword}")
                for post in self.reddit.subreddit(subreddit).search(keyword, limit=limit):
                    results.append({
                        "title": post.title,
                        "url": post.url,
                        "author": self.get_author_info(post.author.name),
                        "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        "subreddit": subreddit,
                        "comments_url": post.permalink,
                        "num_comments": post.num_comments,
                    })
            except Exception as e:
                logging.error(
                    f"Error searching for keyword '{keyword}' in subreddit '{subreddit}': {e}")
        return results

    def fetch_comments(self, permalink):
        """Fetch comments from a Reddit post."""
        try:
            post = self.reddit.submission(
                url=f"https://www.reddit.com{permalink}")
            post.comments.replace_more(limit=0)
            comments = [{
                "id": comment.id,
                "comment": comment.body,
                "author": self.get_author_info(comment.author.name),
                "created_utc": comment.created_utc,
                "permalink": comment.permalink
            } for comment in post.comments.list()]
            logging.info(
                f"Fetched {len(comments)} comments for post: {post.title}")
            return comments
        except Exception as e:
            logging.error(
                f"Error fetching comments for permalink '{permalink}': {e}")
            return []

    def export_to_csv(self, data, filename="reddit_leads.csv"):
        """Export scraped data to a CSV file."""
        if not data:
            logging.warning("No data to export.")
            return
        try:
            keys = data[0].keys()
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
            logging.info(f"Exported {len(data)} records to {filename}")
        except Exception as e:
            logging.error(f"Error exporting data to CSV: {e}")


# if __name__ == "__main__":
#     # Define subreddits and keywords
#     SUBREDDITS = [
#         "Productivity",
#         "RemoteWork",
#         "SaaS",
#         "ProjectManagement",
#         "WorkFromHome",
#         "NoCode",
#     ]
#     KEYWORDS = ["integration", "sync", "Slack",
#                 "Trello", "workflow", "Google Docs"]

#     # Instantiate and run the scraper
#     scraper = RedditScraper(
#         user_agent="SyncLoom Research Script by /u/YOUR_USERNAME",
#     )

#     # for subreddit in SUBREDDITS:
#     #     results = scraper.search_reddit(subreddit, KEYWORDS)

#     #     for result in results:
#     #         comments = scraper.fetch_comments(result.get('comments_url'))
#     #         logging.info(f"comments for subreddit {subreddit} : {comments}")
