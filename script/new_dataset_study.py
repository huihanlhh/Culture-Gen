from es import count_documents_containing_phrases
import os
import json
import csv
import pickle as pkl
from tqdm import tqdm
import argparse
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from elasticsearch import Elasticsearch

def get_nationality_count(home_dir, cache_path, es, index=None):
    """
    Get the count of documents containing nationalities
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        nationalities_tup = [(row[0], row[1]) for row in reader]

    document_cache_path = cache_path.replace(".pkl", "_documents.pkl")
    # search for number of documents containing all nationalities and countries
    if os.path.exists(document_cache_path):
        count_documents = pkl.load(open(document_cache_path, "rb"))
    else:
        count_documents = {}
    for i, (country, nationality) in enumerate(tqdm(nationalities_tup, desc="Counting nationalities")):
        if nationality in count_documents:
            continue
        country = country.lower()
        nationality = nationality.lower()
        count_country = count_documents_containing_phrases(index, country, es=es)
        count_nationality = count_documents_containing_phrases(index, nationality, es=es)
        # Edge cases for Myanmar and Denmark
        if country == "myanmar":
            count_country += count_documents_containing_phrases(index, "burma", es=es)
            count_nationality += count_documents_containing_phrases(index, "burmese", es=es)
        if nationality == "dane":
            count_nationality += count_documents_containing_phrases(index, "danish", es=es)
        count_all = count_documents_containing_phrases(index, [country, nationality], all_phrases=True, es=es)
        if country == "myanmar":
            count_all += count_documents_containing_phrases(index, ["burma", "burmese"], all_phrases=True, es=es)
        if nationality == "dane":
            count_all += count_documents_containing_phrases(index, [country, "danish"], all_phrases=True, es=es)
        count_documents[nationality] = count_country + count_nationality - count_all
        logger.info(f"Counted documents containing {nationality}: {count_documents[nationality]}")

        pkl.dump(count_documents, open(document_cache_path, "wb"))
    return

def get_nationality_topic_count(home_dir, cache_path, es, index=None):
    """
    Get the count of documents containing nationalities and topics
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        nationalities_tup = [(row[0], row[1]) for row in reader]
    # for each topic, we have a list of keywords that refer to this topic
    topic_to_keywords_mapping = {
        "favorite_music": ["music", "song", "songs", "album", "albums", "band", "bands", "singer", "singers", "musician", "musicians", "genre", "genres", "concert", "concerts"],
        "exercise_routine": ["exercise", "routine", "workout", "sport", "sports"],
        "music_instrument": ["music instrument", "music instruments", "instrument", "instruments"],
        "favorite_show_or_movie": ["movie", "movies", "film", "films", "TV show", "TV shows", "TV series", "cinema"],
        "food": ["food", "foods", "cuisine", "cuisines", "dish", "dishes", "meal", "meals", "recipe", "recipes", "menu", "menus", "breakfast", "lunch", "dinner", "snack", "snacks"],
        "picture_on_the_front_door": ["picture", "pictures", "painting", "paintings", "portrait", "portraits"],
        "statue_on_the_front_door": ["statue", "statues", "sculpture", "sculptures"],
        "clothing": ["clothing", "clothes", "apparel", "garment", "garments", "outfit", "outfits", "attire", "attires", "dress", "dresses", "suit", "suits", "uniform", "uniforms"],
    }
    if os.path.exists(cache_path):
        count_documents = pkl.load(open(cache_path, "rb"))
    else:
        count_documents = {}
    for topic, keywords in topic_to_keywords_mapping.items():
        if topic not in count_documents:
            count_documents[topic] = {}
        for keyword in keywords:
            if keyword not in count_documents[topic]:
                count_documents[topic][keyword] = {}
            for i, (country, nationality) in enumerate(tqdm(nationalities_tup, desc=f"Counting {topic}")):
                country = country.lower()
                nationality = nationality.lower()
                if nationality not in count_documents[topic][keyword]:
                    count_country = count_documents_containing_phrases(index, [country, keyword], all_phrases=True, es=es)
                    count_nationality = count_documents_containing_phrases(index, [nationality, keyword], all_phrases=True, es=es)
                    if country == "myanmar":
                        count_country += count_documents_containing_phrases(index, ["burma", keyword], all_phrases=True, es=es)
                        count_nationality += count_documents_containing_phrases(index, ["burmese", keyword], all_phrases=True, es=es)
                    if nationality == "dane":
                        count_nationality += count_documents_containing_phrases(index, ["danish", keyword], all_phrases=True, es=es)
                    count_all = count_documents_containing_phrases(index, [country, nationality, keyword], all_phrases=True, es=es)
                    if country == "myanmar":
                        count_all += count_documents_containing_phrases(index, ["burma", "burmese", keyword], all_phrases=True, es=es)
                    if nationality == "dane":
                        count_all += count_documents_containing_phrases(index, [country, "danish", keyword], all_phrases=True, es=es)
                    count_documents[topic][keyword][nationality] = count_country + count_nationality - count_all
                    logger.info(f"Counted documents containing {keyword} with {nationality}")
                    pkl.dump(count_documents, open(cache_path, "wb"))

def plot_nationality_count(nationality_count_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    count_documents = pkl.load(open(nationality_count_path, "rb"))
    # sort
    count_documents = {k: v for k, v in sorted(count_documents.items(), key=lambda item: item[1], reverse=True)}
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(25, 12))
    sns.barplot(x=list(count_documents.keys()), y=list(count_documents.values()), palette="viridis", label="Nationality Count in Redpajama")
    # remove grid
    plt.grid(False)

    plt.xticks(rotation=90, fontsize=10)
    plt.savefig(f"redpajama_nationality.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--count", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    
    args = parser.parse_args()
    logger.info(args)

    es = None # TODO: create elasticsearch instance
    if args.count == "nationality":
        get_nationality_count(args.home_dir, f"{args.home_dir}/dataset_search/nationality_count.pkl", es=es, index=args.index)
    elif args.count == "topic":
        get_nationality_topic_count(args.home_dir, f"{args.home_dir}/dataset_search/nationality_topic_count.pkl", es=es, index=args.index)

    if args.plot:
        plot_nationality_count(f"{args.home_dir}/dataset_search/nationality_count_documents.pkl")