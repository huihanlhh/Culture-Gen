"""Evaluates the markedness and diversity of cultural generations"""
import json
import pickle as pkl
import csv
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import spearmanr, kendalltau
from symbol_utils import process_generation_to_culture_symbol
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def simpson_iod(counter):

    """ Given a hash { 'species': count } , returns the Simpson Index of Diversity
    
    >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
    0.3888888888888889
    """

    def p(n, N):
        """ Relative abundance """
        if n  == 0:
            return 0
        else:
            return float(n)/N

    N = sum(counter.values())
    
    return 1 - sum(p(n, N)**2 for n in counter.values() if n != 0)


def eval_markedness(home_dir, original_generation_path, save_path, topic_list=None, role="neighbor"):
    """
        If a generation contains:
            - "traditional" "typical" or mention of self-nationality
            - parentheses that explains the generation
        then it is considered marked.

        We evaluate the number of culture-dependent generations that are marked
    """
    with open(original_generation_path, "r") as f:
        topic_nationality_dict = json.load(f)
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        nationalities_tup = [(row[0], row[1]) for row in reader]
        nationalities_tup.append(("",""))
    logger.info("Loaded nationalities")
    if topic_list == None:
        topic_list = list(topic_nationality_dict.keys())
    topic_nationality_count_dict = {}
    for a, topic in enumerate(tqdm(topic_list, desc="counting topics")):
        topic_dict = topic_nationality_dict[topic]
        if topic not in topic_nationality_count_dict:
            topic_nationality_count_dict[topic] = {}
        role_dict = topic_dict[role]
        if role not in topic_nationality_count_dict[topic]:
            topic_nationality_count_dict[topic][role] = {}
        for c, (country, nationality) in enumerate(tqdm(nationalities_tup, desc="counting nationalities")):
            nationality_dict = role_dict[nationality]
            if nationality not in topic_nationality_count_dict[topic][role]:
                topic_nationality_count_dict[topic][role][nationality] = defaultdict(Counter)
            for gender, generations in nationality_dict.items():
                for generation in generations:
                    # ignore uncooperatie answers
                    if "AI" in generation:
                        continue
                    # split by first period (exclude sign, statue and picture)
                    if topic not in ["sign_on_the_front_door", "picture_on_the_front_door", "statue_on_the_front_door"]:
                        generation = generation.split(".")[0]
                    elif topic in ["picture_on_the_front_door", "statue_on_the_front_door"]:
                        values = generation.split(".")
                        if not values[0].endswith("St"):
                            generation = values[0]
                        else:
                            for i, value in enumerate(values[1:]):
                                if not value.endswith("St"):
                                    generation = ".".join(values[0:i+1])
                                    break
                    # if starts with space, remove space
                    generation = generation.strip()
                    # if contains '_', remove
                    generation = generation.replace("_", "")
                    if ("traditional" in generation or "typical" in generation or (nationality != "" and (nationality.lower() in generation.lower() or country.lower() in generation.lower()))) and "(" in generation:
                        topic_nationality_count_dict[topic][role][nationality][gender]["both_mark"] += 1
                        topic_nationality_count_dict[topic][role][nationality][gender]["vocab_mark"] += 0
                        topic_nationality_count_dict[topic][role][nationality][gender]["paren_mark"] += 0
                    elif "traditional" in generation or "typical" in generation or (nationality != "" and (nationality.lower() in generation.lower() or country.lower() in generation.lower())):
                        topic_nationality_count_dict[topic][role][nationality][gender]["vocab_mark"] += 1
                        topic_nationality_count_dict[topic][role][nationality][gender]["paren_mark"] += 0
                        topic_nationality_count_dict[topic][role][nationality][gender]["both_mark"] += 0
                    elif "(" in generation:
                        topic_nationality_count_dict[topic][role][nationality][gender]["paren_mark"] += 1
                        topic_nationality_count_dict[topic][role][nationality][gender]["both_mark"] += 0
                        topic_nationality_count_dict[topic][role][nationality][gender]["vocab_mark"] += 0
                    else:
                        topic_nationality_count_dict[topic][role][nationality][gender]["vocab_mark"] += 0 # just to make sure all keys are present
                        topic_nationality_count_dict[topic][role][nationality][gender]["paren_mark"] += 0 # just to make sure all keys are present
                        topic_nationality_count_dict[topic][role][nationality][gender]["both_mark"] += 0 # just to make sure all keys are present

    with open(save_path, "w") as w:
        json.dump(topic_nationality_count_dict, w, indent=4)

def _plot_world_map_markedness(markedness_data, topic, model, extract="all", gender="", home_dir=None): # extract: all, paren, vocab
    """
        plot markedness data on the world map for a given topic
        extract: all, paren, vocab
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_country_names = world['name'].values

    name_to_gpd_mapping ={
        "Bosnia and Herzegovina":"Bosnia and Herz.",
        "Dominican Republic":"Dominican Rep.",
        "United States":"United States of America",
    }

    world['markedness_data'] = None

    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    for country, nationality in countries_nationalities_list:
        if country not in world_country_names:
            if country in name_to_gpd_mapping:
                new_country = name_to_gpd_mapping[country]
            else:
                continue
        else:
            new_country = country
        markedness_value_dict = markedness_data[nationality][gender]
        if extract == "all":
            markedness_value = float(sum(markedness_value_dict.values()))
        elif extract == "paren":
            markedness_value = float(markedness_value_dict["paren_mark"] + markedness_value_dict["both_mark"])
        elif extract == "vocab":
            markedness_value = float(markedness_value_dict["vocab_mark"] + markedness_value_dict["both_mark"])
        world.loc[world['name'] == new_country, 'markedness_data'] = markedness_value
    # print datatype of the column markedness_data
    world["markedness_data"] = world["markedness_data"].astype(float)
            
    # Plot the world map with heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot the countries
    world.boundary.plot(ax=ax, linewidth=1, color='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Plot the heatmap
    world.plot(column='markedness_data', ax=ax, cmap='OrRd', legend=True, missing_kwds={'color': 'gray'}, cax=cax)

    plt.savefig(f"../markedness_world_map_{topic}_{gender}_{model}_{extract}.png")

def plot_world_map_with_markedness(eval_path, model_name, topic_list=None, gender="", aggregate=False):
    """
        Wrapper function to iteratve over all topic and all extract type

        `aggregate` is a boolean value that indicates whether to aggregate all topics and only generate one plot
    """
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    if not aggregate:
        for topic in topic_list:
            topic_dict = topic_nationality_eval_dict[topic]
            role = "neighbor" # pilot study; expand to more roles in the future
            nationality_counter = topic_dict[role]
            for extract in ["all", "paren", "vocab"]:
                _plot_world_map_markedness(nationality_counter, topic, model_name, extract=extract, gender=gender) # all, paren, vocab
    else:
        # aggregate all topics
        aggregate_dict = {}
        for topic in topic_list:
            topic_dict = topic_nationality_eval_dict[topic]
            role = "neighbor" # pilot study; expand to more roles in the future
            nationality_counter = topic_dict[role]
            for nationality, nationality_dict in nationality_counter.items():
                if nationality not in aggregate_dict:
                    aggregate_dict[nationality] = {gender: Counter()}
                markedness_value_dict = nationality_dict[gender]
                for key, value in markedness_value_dict.items():
                    aggregate_dict[nationality][gender][key] += value
        # normalize by the number of topics
        for nationality, nationality_dict in aggregate_dict.items():
            for key, value in nationality_dict[gender].items():
                aggregate_dict[nationality][gender][key] = value / len(topic_list)
        for extract in ["all", "paren", "vocab"]:
            _plot_world_map_markedness(aggregate_dict, "aggregate", model_name, extract=extract, gender=gender) # all, paren, vocab


def plot_bar_chart_with_markedness(eval_path, model_name, eval_type="markedness", topic_list=None, gender="", secondary_ax_path=None, secondary=None, mark_type="vocab"): # mark_type="paren"
    """
        plot barchart for each topic on markedness
        secondary: nationality, topic, or ratio of the two. Serves a the secondary axis in the bar plot
        secondary_ax_path: path to the secondary axis data
        mark_type: vocab or paren, serves as the type of markedness to plot
    """
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        topic_dict = topic_nationality_eval_dict[topic]
        role = "neighbor" # pilot study; expand to more roles in the future
        nationality_counter = topic_dict[role]

        del nationality_counter[""]
        keys = list(nationality_counter.keys())
        all_tuples = [(key, nationality_counter[key][gender]) for key in keys] # string: Dict[str, int]

        ordered_tuples = sorted(all_tuples, key=lambda x: x[1]["both_mark"] + x[1][f"{mark_type}_mark"], reverse=True)
        if secondary == "nationality":
            with open(secondary_ax_path, "rb") as r:
                secondary_dict = pkl.load(r)
            secondary_tuples = [(key, secondary_dict[key.lower()]) for key, v in ordered_tuples]
        elif secondary == "topic":
            with open(secondary_ax_path, "rb") as r:
                secondary_dict = pkl.load(r)[topic]
            secondary_tuples = [(key, sum([d[key.lower()] for d in secondary_dict.values()])) for key, v in ordered_tuples]
        
        nationalities, count_dict = zip(*ordered_tuples)
        nationalites_arr = np.array(nationalities)

        marks = np.array([d["both_mark"] + d[f"{mark_type}_mark"] for d in count_dict])
        secondary_values_arr = np.array([tup[1] for tup in secondary_tuples])

        fig, ax1 = plt.subplots(figsize=(25, 12))
        sns.set(style="whitegrid")

        sns.barplot(x=nationalites_arr, y=marks, ax=ax1, color="teal", alpha=0.8, label=eval_type)
        ax2 = ax1.twinx()
        sns.barplot(x=nationalites_arr, y=secondary_values_arr, ax=ax2, color="salmon", alpha=0.8, label='frequency')
        ax2.invert_yaxis()

        # hide top and right spine of ax1
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # Remove the grid
        ax1.grid(False)
        ax2.grid(False)

        # Add legends outside of the figure
        ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 0.95))
        
        ax1.spines['left'].set_color('teal')  # Change the color of the left y-axis
        ax2.spines['right'].set_color('salmon')  # Change the color of the right y-axis
        ax1.tick_params(axis='y', colors='teal')  # Change the text color of the left y-axis labels
        ax2.tick_params(axis='y', colors='salmon')   # Change the text color of the right y-axis labels

        ax1.set_xticks(ticks=range(len(nationalities)), labels=nationalities, fontsize=10, rotation=90)
        plt.savefig(f"../{eval_type}_{topic}_{gender}_{model_name}_{mark_type}.png")
        plt.clf()

def plot_bar_chart_with_markedness_by_topic(eval_paths, model_names, topic_list, gender=""):
    """
        Plot the overall average marked generations for each model by topic
    """
    model_data = defaultdict(list)
    all_topic_list = []
    for i, eval_path in enumerate(eval_paths):
        with open(eval_path, "r") as r:
            topic_nationality_eval_dict = json.load(r)
        if topic_list == None:
            topic_list = list(topic_nationality_eval_dict.keys())
        all_topic_list = topic_list
        for topic in topic_list:
            topic_dict = topic_nationality_eval_dict[topic]
            role = "neighbor" # pilot study; expand to more roles in the future
            nationality_counter = topic_dict[role]

            nationality_counter["null"] = nationality_counter[""]
            del nationality_counter[""]
            keys = list(nationality_counter.keys())
            all_tuples = [(key, nationality_counter[key][gender]) for key in keys] # string: Dict[str, int]
            ordered_tuples = sorted(all_tuples, key=lambda x: sum(x[1].values()), reverse=True)
            average_markedness = sum([sum(d[1].values()) for d in ordered_tuples]) / len(ordered_tuples)
            model_data[model_names[i]].append(average_markedness)
    model_data["Topic"] = all_topic_list
    print(model_data)
    df = pd.DataFrame(model_data)
    # Melt the dataframe to long format
    df_melted = pd.melt(df, id_vars='Topic', var_name='Model', value_name='Marked Generations')

    # Plot
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Topic', y='Marked Generations', hue='Model', palette="Dark2", data=df_melted)
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100
    plt.xlabel(None) # do not plot x-axis label

    plt.ylabel('Average Marked Generations')
    plt.savefig(f"../markedness_bar_by_topic.png")

def plot_continentwise_markedness(home_dir, eval_paths, model_names, topic_list, gender=""):
    """
        Plot the average markedness for each continent for each topic in one bar chart, for each model
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_group_list = [(row[0], row[1], row[2]) for row in reader]
    logger.info("Loaded nationalities")
    topic_model_average_dict = {}
    categories = ["African-Islamic", "Baltic", "Central-Asian", "East-Asian", "Eastern-European", "Latin-American", "Nordic", "South-Asian", "Southeast-Asian", "Western-European", "English-Speaking", "Middle-Eastern"]
    
    # for each topic, each model, each continent, calculate the average markedness
    for eval_path, model_name in zip(eval_paths, model_names):
        with open(eval_path, "r") as r:
            topic_nationality_eval_dict = json.load(r)
        if topic_list == None:
            topic_list = list(topic_nationality_eval_dict.keys())
        topic_model_average_dict[model_name] = {}
        for topic in topic_list:
            topic_model_average_dict[model_name][topic] = defaultdict(list)
            topic_dict = topic_nationality_eval_dict[topic]
            role = "neighbor" # pilot study; expand to more roles in the future
            nationality_counter = topic_dict[role]

            nationality_counter["null"] = nationality_counter[""]
            del nationality_counter[""]
            for country, nationality, grouping in countries_nationalities_group_list:
                markedness = sum(nationality_counter[nationality][gender].values())
                topic_model_average_dict[model_name][topic][categories[int(grouping)]].append(markedness)

    # calculate the average markedness for each continent for each topic
    continent_model_average_dict = {}
    for model_name, topic_dict in topic_model_average_dict.items():
        continent_model_average_dict[model_name] = {}
        for topic, continent_dict in topic_dict.items():
            for continent, markedness_list in continent_dict.items():
                average_markedness = sum(markedness_list) / len(markedness_list)
                if continent not in continent_model_average_dict[model_name]:
                    continent_model_average_dict[model_name][continent] = {}
                continent_model_average_dict[model_name][continent][topic] = average_markedness
    
    # plot the average markedness for each continent for each topic in one bar chart
    for model_name, continent_dict in continent_model_average_dict.items():
        model_data = defaultdict(list)
        for continent, topic_dict in continent_dict.items():
            for topic, average_markedness in topic_dict.items():
                model_data[topic].append(average_markedness)
        model_data["Continent"] = list(continent_dict.keys())
        df = pd.DataFrame(model_data)
        df_melted = pd.melt(df, id_vars='Continent', var_name='Topic', value_name='Marked Generations')
        plt.figure(figsize=(25, 6))
        sns.barplot(x='Topic', y='Marked Generations', hue='Continent', palette="Spectral", data=df_melted)
        plt.ylim(0, 100)  # Set y-axis limit from 0 to 100
        plt.xlabel(None)
        plt.ylabel('Average Marked Generations')
        if model_name == "gpt-4":
            plt.legend(title="Geographic Regions", loc='upper right', bbox_to_anchor=(1.1, 1))
        else:
            plt.legend(title="Geographic Regions", loc='upper right')
        plt.savefig(f"../continentwise_markedness_{model_name}.png")

def eval_skewness(home_dir, new_shortened_path, cache_dict_path, culture_symbol_path, save_path, topic_list=None, role="neighbor", gender=""):
    """
        We evaluate the strength (weighted percentage) or skewness (percentage) of generations that contains the cultural symbols of a certain culture
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(new_shortened_path, "r") as r:
        category_nationality_dict = json.load(r)

    if os.path.exists(save_path):
        with open(save_path, "r") as r:
            strength_skewness_dict = json.load(r)
    else:
        strength_skewness_dict = {}

    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    for topic in topic_list:
        topic_cache_dict_path = cache_dict_path.replace(".pkl", f"_{topic}.pkl")
        topic_culture_symbol_path = culture_symbol_path.replace(".json", f"_{topic}.json")
        if not os.path.exists(topic_cache_dict_path) or not os.path.exists(topic_culture_symbol_path):
            continue
        with open(topic_cache_dict_path, "rb") as r:
            cache_dict = pkl.load(r)
        with open(topic_culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)
        if topic not in strength_skewness_dict:
            strength_skewness_dict[topic] = {}
        if gender not in strength_skewness_dict[topic]:
            strength_skewness_dict[topic][gender] = {}
            for nationality_index, (country, nationality) in enumerate(countries_nationalities_list):
                generated_values = category_nationality_dict[topic][role][nationality][gender]
                all_scores = []
                for i, value in enumerate(tqdm(generated_values, desc="calculating culture symbol strength scores for each generation")):
                    # process the shortened generations the same way as extracting symbols
                    if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
                        continue
                    all_values = []
                    for phrase in value.split(";"):
                        # remove marked expressions
                        if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or nationality in phrase or country in phrase:
                            continue
                        phrase = process_generation_to_culture_symbol(phrase)
                        if phrase.strip() != "":
                            all_tokens = phrase.lower().split()
                            # find all unigrams
                            unigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token)) for token in all_tokens]
                            bigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2]))) for i in range(len(all_tokens)-1)]
                            trigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3]))) for i in range(len(all_tokens)-2)]
                            fourgrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4]))) for i in range(len(all_tokens)-3)]
                            # rank the probability of each ngram
                            ngrams = unigrams + bigrams + trigrams + fourgrams
                            probabilities = [cache_dict[ngram][nationality_index] for ngram in ngrams]
                            tups = list(zip(ngrams, probabilities))
                            tups = sorted(tups, key=lambda x: x[1], reverse=True)
                        else:
                            continue
                        if tups is not None:
                            phrase_representative_value = tups[0][0]
                            all_values.append(phrase_representative_value) # TODO: change into appending all tups and select the one with highest culture_score later
                    # calculate the sum of culture symbol strengths for this generation
                    culture_scores = 0
                    for phrase_representative_value in all_values:
                        if phrase_representative_value in culture_symbol_dict and nationality in culture_symbol_dict[phrase_representative_value]:
                            # the culture score is the inverse of the number of nationalities that the phrase is culture symbol of
                            culture_score = 1 / len(culture_symbol_dict[phrase_representative_value])
                            culture_scores += culture_score
                    all_scores.append(culture_scores)
                # calculate the skewness of the culture symbol strength scores
                strength = sum(all_scores) / len(all_scores)
                skewness = len([score for score in all_scores if score > 0]) / 100
                strength_skewness_dict[topic][gender][nationality] = (strength, skewness)
    with open(save_path, "w") as w:
        json.dump(strength_skewness_dict, w, indent=4)

def _plot_world_map_skewness(skewness_data, topic, model, extract="skewness", gender="", home_dir=None): # strength, skewness
    """
        plot skewness data on the world map for a given topic
        extract: strength or skewness
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    world_country_names = world['name'].values

    name_to_gpd_mapping ={
        "Bosnia and Herzegovina":"Bosnia and Herz.",
        "Dominican Republic":"Dominican Rep.",
        "United States":"United States of America",
    }

    world['skewness_data'] = None

    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    for country, nationality in countries_nationalities_list:
        if country not in world_country_names:
            if country in name_to_gpd_mapping:
                new_country = name_to_gpd_mapping[country]
            else:
                continue
        else:
            new_country = country
        skewness_value_list = skewness_data[gender][nationality]
        if extract == "strength":
            skewness_value = float(skewness_value_list[0])
        elif extract == "skewness":
            skewness_value = float(skewness_value_list[1])
        world.loc[world['name'] == new_country, 'skewness_data'] = skewness_value
    # print datatype of the column markedness_data
    world["skewness_data"] = world["skewness_data"].astype(float)

    # Plot the world map with heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot the countries
    world.boundary.plot(ax=ax, linewidth=1, color='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Plot the heatmap
    world.plot(column='skewness_data', ax=ax, cmap='OrRd', legend=True, missing_kwds={'color': 'gray'}, cax=cax)

    plt.savefig(f"../skewness_world_map_{topic}_{gender}_{model}_{extract}.png")

def plot_world_map_with_skewness(eval_path, model_name, topic_list=None, gender=""):
    """
        Wrapper function to iteratve over all topic and all extract type
    """
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        topic_dict = topic_nationality_eval_dict[topic]
        nationality_counter = topic_dict
        for extract in ["strength", "skewness"]:
            _plot_world_map_skewness(nationality_counter, topic, model_name, extract=extract, gender=gender) # strength, skewness

def eval_gpt4_diversity(home_dir, new_shortened_path, culture_symbol_path, save_path, topic_list=None, role="neighbor", gender=""):
    """
        Evaluate diversity without culture symbol assignment process
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(new_shortened_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r:
            symbol_counter_dict = json.load(r)
    else:
        symbol_counter_dict = {}
    
    with open(culture_symbol_path, "r") as r:
        culture_symbol_dict = json.load(r)
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())

    # if phrases are one of the keywords then ignore
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
    for topic in topic_list:
        if topic not in symbol_counter_dict:
            symbol_counter_dict[topic] = {}
        if gender not in symbol_counter_dict[topic]:
            symbol_counter_dict[topic][gender] = {}
            for nationality_index, (country, nationality) in enumerate(countries_nationalities_list):
                generated_values = category_nationality_dict[topic][role][nationality][gender]
                culture_symbol_counter = Counter()
                for i, value in enumerate(tqdm(generated_values, desc="calculating culture symbol strength scores for each generation")):
                    # process the shortened generations the same way as extracting symbols
                    if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
                        continue
                    all_values = []
                    for phrase in value.split(";"):
                        # remove marked expressions
                        if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or nationality in phrase or country in phrase:
                            continue
                        phrase = process_generation_to_culture_symbol(phrase)
                        # remove the last token if length <= 2
                        if phrase.strip() != "" and len(phrase.split()[-1]) <= 2:
                            phrase = " ".join(phrase.split()[:-1])
                        for symbol in culture_symbol_dict[topic]:
                            if symbol.lower() + " " in phrase.lower() + " ":
                                all_values.append(symbol)
                                break
                    # add each value to the dict if previous values are not a token in the current value
                    for phrase_representative_value in all_values:
                        if phrase_representative_value in topic_to_keywords_mapping[topic]:
                            continue
                        culture_symbol_counter[phrase_representative_value] += 1
                # calculate the simpson index
                simpson_iod_value = simpson_iod(culture_symbol_counter)
                symbol_counter_dict[topic][gender][nationality] = (simpson_iod_value, len(culture_symbol_counter))
    with open(save_path, "w") as w:
        json.dump(symbol_counter_dict, w, indent=4)
        

def eval_diversity(home_dir, new_shortened_path, cache_dict_path, culture_symbol_path, save_path, topic_list=None, role="neighbor", gender=""):
    """
        We evaluate the number (and simpson index - deprecated) of unique cultural symbols in the generation of a certain culture
        
        For a generation containing multiple possible cultural symbols, we count all symbols allowing duplication
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(new_shortened_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r:
            symbol_counter_dict = json.load(r)
    else:
        symbol_counter_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())

    # if phrases are one of the keywords then ignore
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
    for topic in topic_list:
        topic_cache_dict_path = cache_dict_path.replace(".pkl", f"_{topic}.pkl")
        topic_culture_symbol_path = culture_symbol_path.replace(".json", f"_{topic}.json")
        if not os.path.exists(topic_cache_dict_path) or not os.path.exists(topic_culture_symbol_path):
            continue
        with open(topic_cache_dict_path, "rb") as r:
            cache_dict = pkl.load(r)
        with open(topic_culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)
        if topic not in symbol_counter_dict:
            symbol_counter_dict[topic] = {}
        if gender not in symbol_counter_dict[topic]:
            symbol_counter_dict[topic][gender] = {}
            for nationality_index, (country, nationality) in enumerate(countries_nationalities_list):
                generated_values = category_nationality_dict[topic][role][nationality][gender]
                culture_symbol_counter = Counter()
                for i, value in enumerate(tqdm(generated_values, desc="calculating culture symbol strength scores for each generation")):
                    # process the shortened generations the same way as extracting symbols
                    if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
                        continue
                    all_values = []
                    for phrase in value.split(";"):
                        # remove marked expressions
                        if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or nationality in phrase or country in phrase:
                            continue
                        phrase = process_generation_to_culture_symbol(phrase)
                        if phrase.strip() != "":
                            all_tokens = phrase.lower().split()
                            # find all unigrams
                            unigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token)) for token in all_tokens]
                            bigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2]))) for i in range(len(all_tokens)-1)]
                            trigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3]))) for i in range(len(all_tokens)-2)]
                            fourgrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4]))) for i in range(len(all_tokens)-3)]
                            # rank the probability of each ngram
                            ngrams = unigrams + bigrams + trigrams + fourgrams
                            probabilities = [cache_dict[ngram][nationality_index] for ngram in ngrams]
                            tups = list(zip(ngrams, probabilities))
                            tups = sorted(tups, key=lambda x: x[1], reverse=True)
                        else:
                            continue
                        if tups is not None:
                            phrase_representative_value = tups[0][0]
                            all_values.append(phrase_representative_value) # TODO: change into appending all tups and select the one with highest culture_score later
                    # for each value, add to counter
                    for phrase_representative_value in all_values:
                        if phrase_representative_value in topic_to_keywords_mapping[topic]:
                            continue
                        if phrase_representative_value in culture_symbol_dict and nationality in culture_symbol_dict[phrase_representative_value]:
                            culture_symbol_counter[phrase_representative_value] += 1
                            # calculate the simpson index
                simpson_iod_value = simpson_iod(culture_symbol_counter)
                symbol_counter_dict[topic][gender][nationality] = (simpson_iod_value, len(culture_symbol_counter))
    with open(save_path, "w") as w:
        json.dump(symbol_counter_dict, w, indent=4)

def _plot_world_map_diversity(diversity_data, topic, model, extract="diversity", gender="", home_dir=None):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    world_country_names = world['name'].values

    name_to_gpd_mapping ={
        "Bosnia and Herzegovina":"Bosnia and Herz.",
        "Dominican Republic":"Dominican Rep.",
        "United States":"United States of America",
    }

    world['diversity_data'] = None

    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]

    for country, nationality in countries_nationalities_list:
        if country not in world_country_names:
            if country in name_to_gpd_mapping:
                new_country = name_to_gpd_mapping[country]
            else:
                continue
        else:
            new_country = country
        if extract == "simpson":
            diversity_value = diversity_data[gender][nationality][0]
            if diversity_value == 1:
                diversity_value = 0
            else:
                diversity_value += 0.01
        elif extract == "count":
            diversity_value = diversity_data[gender][nationality][1]
        world.loc[world['name'] == new_country, 'diversity_data'] = diversity_value
    # print datatype of the column diversity_data
    world["diversity_data"] = world["diversity_data"].astype(float)
            

    # Plot the world map with heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot the countries
    world.boundary.plot(ax=ax, linewidth=1, color='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Plot the heatmap
    world.plot(column='diversity_data', ax=ax, cmap='OrRd', legend=True, missing_kwds={'color': 'gray'}, cax=cax)

    plt.savefig(f"../diversity_world_map_{topic}_{gender}_{model}_{extract}.png")

def plot_world_map_with_diversity(eval_path, model_name, topic_list=None, gender=""):
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        topic_dict = topic_nationality_eval_dict[topic]
        nationality_counter = topic_dict
        for extract in ["simpson", "count"]:
            _plot_world_map_diversity(nationality_counter, topic, model_name, extract=extract, gender=gender)

def plot_bar_chart_with_diversity(home_dir, eval_path, model_name, eval_type="simpson", topic_list=None, gender="", secondary_ax_path=None, secondary=None): # or eval_type=skewness
    """
        Plot the diversity of generations with either count of simpson index
        secondary: nationality or topic count in training data, as the secondary axis
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1], row[2]) for row in reader]
    grouping_symbol_counter_dict = defaultdict(Counter)
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        if topic not in topic_nationality_eval_dict:
            continue
        topic_dict = topic_nationality_eval_dict[topic][gender]
        if eval_type == "simpson":
            i = 0
        else:
            i = 1
        for key, value in topic_dict.items():
            if value[0] == 1:
                topic_dict[key][0] = 0
            else:
                topic_dict[key][0] += 0.01
        for country, nationality, grouping in countries_nationalities_list:
            grouping_symbol_counter_dict[grouping][nationality] = topic_dict[nationality][i]
        
        fig, ax1 = plt.subplots(figsize=(30, 12))
        ax2 = ax1.twinx()
        ax2.invert_yaxis()
        sns.set(style="whitegrid")
        x_labels = []
        y_values = []
        secondary_y_values = []
        for i, (group, nationality_counter) in enumerate(list(grouping_symbol_counter_dict.items())):
            ordered_tuples = sorted(nationality_counter.items(), key=lambda x: x[1], reverse=True)
            if secondary == "nationality":
                with open(secondary_ax_path, "rb") as r:
                    secondary_dict = pkl.load(r)
                secondary_tuples = [(key, secondary_dict[key.lower()]) for key, v in ordered_tuples]
            elif secondary == "topic":
                with open(secondary_ax_path, "rb") as r:
                    secondary_dict = pkl.load(r)[topic]
                secondary_tuples = [(key, sum([d[key.lower()] for d in secondary_dict.values()])) for key, v in ordered_tuples]
            nationalities, values = zip(*ordered_tuples)

            values_arr = np.array(values)
            secondary_values_arr = np.array([tup[1] for tup in secondary_tuples])
            x_labels.extend(nationalities)
            x_labels.append(" ")
            x_labels.append(" ") 
            y_values.extend(values_arr)
            y_values.append(0)
            y_values.append(0)
            secondary_y_values.extend(secondary_values_arr)
            secondary_y_values.append(0)
            secondary_y_values.append(0)

        ax1.bar(x=range(len(x_labels)), height=y_values, color="teal", alpha=0.8, label="Count")
        ax2.bar(x=range(len(x_labels)), height=secondary_y_values, color="salmon", alpha=0.8, label='Frequency')

        ax1.set_ylim(0, max(y_values) + 1)
        ax1.set_xticks(ticks=range(len(x_labels)),labels=x_labels, rotation=90,fontsize=8)
        ax1.legend(loc='lower right')
        ax2.legend(loc='upper right')
        plt.tight_layout()

        # hide top and right spine of ax1
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # # Remove the grid
        ax1.grid(False)
        ax2.grid(False)
        

        ax1.spines['left'].set_color('teal')  # Change the color of the left y-axis
        ax2.spines['right'].set_color('salmon')  # Change the color of the right y-axis
        ax1.tick_params(axis='y', colors='teal')  # Change the text color of the left y-axis labels
        ax2.tick_params(axis='y', colors='salmon')   # Change the text color of the right y-axis labels
        plt.savefig(f"../{eval_type}_{topic}_{gender}_{model_name}_{secondary}_grouped.png")
        plt.clf()

def calculate_diversity_correlation_with_training_data(eval_path, eval_type="count", topic_list=None, gender="", home_dir=None):
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        print(topic)
        if topic not in topic_nationality_eval_dict:
            continue
        topic_dict = topic_nationality_eval_dict[topic][gender]
        for key, value in topic_dict.items():
            if value[0] == 1:
                topic_dict[key][0] = 0
            else:
                topic_dict[key][0] += 0.01
        all_tuples = list(topic_dict.items())
        if eval_type == "simpson":
            i = 0
        else:
            i = 1
        ordered_tuples = sorted(all_tuples, key=lambda x: x[1][i], reverse=True)
        secondary_path_n= f"{home_dir}/dataset_search/nationality_count_documents.pkl"
        secondary_path_t = f"{home_dir}/dataset_search/nationality_topic_count.pkl"
        with open(secondary_path_n, "rb") as r:
            secondary_dict_n = pkl.load(r)
        with open(secondary_path_t, "rb") as r:
            secondary_dict_t = pkl.load(r)[topic]
        secondary_tuples_n = [(key, secondary_dict_n[key.lower()]) for key, v in ordered_tuples]
        secondary_tuples_t = [(key, sum([d[key.lower()] for d in secondary_dict_t.values()])) for key, v in ordered_tuples]
        
        nationalities, values = zip(*ordered_tuples)
        values_arr = [tup[i] for tup in values]
        secondary_values_arr_n =[tup[1] for tup in secondary_tuples_n]
        secondary_values_arr_t =[tup[1] for tup in secondary_tuples_t]
        # calculate spearman correlation and kendall tau
        spearman_corr_n = spearmanr(values_arr, secondary_values_arr_n)
        kendall_tau_n = kendalltau(values_arr, secondary_values_arr_n)
        spearman_corr_t = spearmanr(values_arr, secondary_values_arr_t)
        kendall_tau_t = kendalltau(values_arr, secondary_values_arr_t)
        print("correlation with nationality count:", spearman_corr_n, kendall_tau_n)
        print("correlation with topic count:",spearman_corr_t, kendall_tau_t)
        print("correlation of n with t")
        print(spearmanr(secondary_values_arr_n, secondary_values_arr_t))
        print(kendalltau(secondary_values_arr_n, secondary_values_arr_t))

def calculate_markedness_correlation_with_training_data(eval_path, eval_type="vocab_mark", topic_list=None, gender="", home_dir=None):
    with open(eval_path, "r") as r:
        topic_nationality_eval_dict = json.load(r)
    if topic_list == None:
        topic_list = list(topic_nationality_eval_dict.keys())
    for topic in topic_list:
        print(topic)
        if topic not in topic_nationality_eval_dict:
            continue
        topic_dict = topic_nationality_eval_dict[topic]["neighbor"]
        all_tuples = [tup for tup in list(topic_dict.items()) if tup[0] != '']
        ordered_tuples = sorted(all_tuples, key=lambda x: x[1][gender]["paren_mark"] + x[1][gender]["both_mark"], reverse=True)
        secondary_path_n= f"{home_dir}/dataset_search/nationality_count_documents.pkl"
        secondary_path_t = f"{home_dir}/dataset_search/nationality_topic_count.pkl"
        with open(secondary_path_n, "rb") as r:
            secondary_dict_n = pkl.load(r)
        with open(secondary_path_t, "rb") as r:
            secondary_dict_t = pkl.load(r)[topic]
        secondary_tuples_n = [(key, secondary_dict_n[key.lower()]) for key, v in ordered_tuples]
        secondary_tuples_t = [(key, sum([d[key.lower()] for d in secondary_dict_t.values()])) for key, v in ordered_tuples]
        
        nationalities, values = zip(*ordered_tuples)

        values_arr = [value[gender][eval_type] for value in values]
        secondary_values_arr_n =[tup[1] for tup in secondary_tuples_n]
        secondary_values_arr_t =[tup[1] for tup in secondary_tuples_t]
        # calculate spearman correlation and kendall tau
        spearman_corr_n = spearmanr(values_arr, secondary_values_arr_n)
        kendall_tau_n = kendalltau(values_arr, secondary_values_arr_n)
        spearman_corr_t = spearmanr(values_arr, secondary_values_arr_t)
        kendall_tau_t = kendalltau(values_arr, secondary_values_arr_t)
        print("correlation with nationality count:", spearman_corr_n, kendall_tau_n)
        print("correlation with topic count:",spearman_corr_t, kendall_tau_t)
        print("correlation of n with t")
        print(spearmanr(secondary_values_arr_n, secondary_values_arr_t))
        print(kendalltau(secondary_values_arr_n, secondary_values_arr_t))

def eval_culture_symbol_presence_in_culture_neutral_prompt(home_dir, new_shortened_path, culture_symbol_path, save_path, topic_list=None, role="neighbor", gender=""):
    """
        We count the number of cultural symbols for each culture that exist in culture neutral prompts
    """
    # Step 1: for each topic, list symbols that are present in the cultural neutral generations
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(new_shortened_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r:
            symbol_counter_dict = json.load(r)
    else:
        symbol_counter_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    for topic in topic_list:
        topic_culture_symbol_path = culture_symbol_path.replace(".json", f"_{topic}.json")
        if not os.path.exists(topic_culture_symbol_path):
            continue
        with open(topic_culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)
        if topic not in symbol_counter_dict:
            symbol_counter_dict[topic] = {}
        if gender not in symbol_counter_dict[topic]:
            symbol_counter_dict[topic][gender] = defaultdict(Counter)
        generated_values = category_nationality_dict[topic][role][""][gender]
        for value in generated_values:
            if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
                continue

            for phrase in value.split(";"):
                # remove marked expressions
                if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase:
                    continue
                phrase = process_generation_to_culture_symbol(phrase)
                if phrase.strip() != "":
                    all_tokens = phrase.lower().split()
                    # find all unigrams
                    unigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token)) for token in all_tokens]
                    bigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2]))) for i in range(len(all_tokens)-1)]
                    trigrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3]))) for i in range(len(all_tokens)-2)]
                    fourgrams = [''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4]))) for i in range(len(all_tokens)-3)]
                    ngrams = unigrams + bigrams + trigrams + fourgrams
                    # find all ngrams that are culture symbols
                    for ngram in ngrams:
                        if ngram in culture_symbol_dict:
                            included_nationalities = culture_symbol_dict[ngram]
                            for n in included_nationalities:
                                symbol_counter_dict[topic][gender][n][ngram] += 1
        with open(save_path, "w") as w:
            json.dump(symbol_counter_dict, w, indent=4)

def eval_gpt4_culture_symbol_presence_in_culture_neutral_prompt(home_dir, new_shortened_path, culture_symbol_path, save_path, topic_list=None, role="neighbor", gender=""):
    """
        We count the number of cultural symbols for each culture that exist in culture neutral prompts, without the culture symbol assignment process
    """
    # Step 1: for each topic, list symbols that are present in the cultural neutral generations
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(new_shortened_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r:
            symbol_counter_dict = json.load(r)
    else:
        symbol_counter_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    for topic in topic_list:
        topic_culture_symbol_path = culture_symbol_path.replace(".json", f"_{topic}.json")
        if not os.path.exists(topic_culture_symbol_path):
            continue
        with open(topic_culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)
        if topic not in symbol_counter_dict:
            symbol_counter_dict[topic] = {}
        if gender not in symbol_counter_dict[topic]:
            symbol_counter_dict[topic][gender] = defaultdict(Counter)
        generated_values = category_nationality_dict[topic][role][""][gender]
        for value in generated_values:
            if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
                continue

            for phrase in value.split(";"):
                # remove marked expressions
                if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase:
                    continue
                phrase = process_generation_to_culture_symbol(phrase)
                if phrase.strip() != "" and len(phrase.split()[-1]) <= 2:
                    phrase = " ".join(phrase.split()[:-1])
                    # find all ngrams that are culture symbols
                for symbol in culture_symbol_dict:
                    if symbol.lower() + " " in phrase.lower() + " ":
                        included_nationalities = culture_symbol_dict[symbol]
                        for n in included_nationalities:
                            symbol_counter_dict[topic][gender][n][symbol] += 1
        with open(save_path, "w") as w:
            json.dump(symbol_counter_dict, w, indent=4)

    return

def remove_culture_symbols_generated_in_agnostic_prompts(culture_symbol_path, culture_agnostic_path, filtered_culture_symbol_path, gender=""):
    """
        We remove culture symbols that are generated in agnostic prompts when counting the culture symbols for each culture
    """
    with open(culture_agnostic_path, "r") as r:
        culture_agnostic_symbol_dict = json.load(r)
    topic_list = culture_agnostic_symbol_dict.keys()
    for topic in topic_list:
        topic_culture_symbol_path = culture_symbol_path.replace(".json", f"_{topic}.json")
        topic_filtered_culture_symbol_path = filtered_culture_symbol_path.replace(".json", f"_{topic}.json")
        if not os.path.exists(topic_culture_symbol_path):
            continue
        with open(topic_culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)
        topic_culture_agnostic_dict = culture_agnostic_symbol_dict[topic][gender]
        new_topic_culture_agnostic_dict = {}
        for symbol in culture_symbol_dict.keys():
            is_in = False
            for nationality, nationality_dict in topic_culture_agnostic_dict.items():
                if symbol in nationality_dict:
                    is_in = True
                    break
            if not is_in:
                new_topic_culture_agnostic_dict[symbol] = culture_symbol_dict[symbol]
        with open(topic_filtered_culture_symbol_path, "w") as w:
            json.dump(new_topic_culture_agnostic_dict, w, indent=4)

def plot_culture_symbols_generated_in_agnostic_prompts(home_dir, agnostic_symbol_path, culture_symbol_prefix, model_name, gender="", secondary_ax_path=None, secondary=None):
    """
        barplot shows the average, while boxplot shows the variance of the number of culture symbols for each culture that is generated in culture agnostic prompts, for each geographic region
    """
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
    
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1], row[2]) for row in reader]
    logger.info("Loaded nationalities")
    with open(agnostic_symbol_path, "r") as r:
        symbol_counter_dict = json.load(r)
    topic_list = symbol_counter_dict.keys()

    for topic in topic_list:
        culture_symbol_path = culture_symbol_prefix.replace(".json", f"_{topic}.json")
        with open(culture_symbol_path, "r") as r:
            culture_symbol_dict = json.load(r)

        grouping_symbol_counter_dict = defaultdict(Counter)
        topic_dict = symbol_counter_dict[topic][gender]

        for country, nationality, grouping in countries_nationalities_list:
            if nationality in topic_dict:
                symbols = topic_dict[nationality]
                symbols = [symbol for symbol in symbols if symbol not in topic_to_keywords_mapping[topic]]
                grouping_symbol_counter_dict[grouping][nationality] = len(symbols)
            else:
                grouping_symbol_counter_dict[grouping][nationality] += 0
        
        categories = ["African-Islamic", "Baltic", "Central-Asian", "East-Asian", "Eastern-European", "Latin-American", "Nordic", "South-Asian", "Southeast-Asian", "Western-European", "English-Speaking", "Middle-Eastern"]

        fig, ax1 = plt.subplots(figsize=(30, 15))

        sns.set(style="whitegrid")
        x_labels = []
        y_values = []
        for i, (group, nationality_counter) in enumerate(list(grouping_symbol_counter_dict.items())):
            ordered_tuples = sorted(nationality_counter.items(), key=lambda x: x[1], reverse=True)
            if secondary == "nationality":
                with open(secondary_ax_path, "rb") as r:
                    secondary_dict = pkl.load(r)
                secondary_tuples = [(key, secondary_dict[key.lower()]) for key, v in ordered_tuples]
            elif secondary == "topic":
                with open(secondary_ax_path, "rb") as r:
                    secondary_dict = pkl.load(r)[topic]
                secondary_tuples = [(key, sum([d[key.lower()] for d in secondary_dict.values()])) for key, v in ordered_tuples]
            nationalities, values = zip(*ordered_tuples)

            values_arr = np.array(values)
            culture_symbol_counts_arr = []
            for n in nationalities:
                counts = 0
                for v in culture_symbol_dict:
                    if v not in topic_to_keywords_mapping[topic] and n in culture_symbol_dict[v]:
                        counts+= 1
                culture_symbol_counts_arr.append(counts)

            x_labels.append(categories[int(group)])
            y_values.append([values_arr[i] / culture_symbol_counts_arr[i] if culture_symbol_counts_arr[i] > 0 else 0 for i in range(len(values_arr))])

        ax1.bar(x=range(len(x_labels)), height=[np.mean(y) for y in  y_values], color="teal", alpha=0.5, label="Culture Agnostic Symbols Overlapping with Geographic Region")
        ax1.boxplot(y_values, positions=range(len(x_labels)), showfliers=True, patch_artist=True, showmeans=True, boxprops=dict(facecolor="teal", color="teal"), medianprops=dict(color="black"), whiskerprops=dict(color="black"), capprops=dict(color="black"))

        ax1.set_ylim(-0.02,1)
        
        ax1.legend(loc='best')

        # Remove the grid
        ax1.grid(False)
        ax1.spines['left'].set_color('teal')  # Change the color of the left y-axis
        ax1.tick_params(axis='y', colors='teal')  # Change the text color of the left y-axis labels
        ax1.set_xticks(ticks=range(len(x_labels)),labels=x_labels, fontsize=25, rotation=20)
        plt.savefig(f"../agnostic_{topic}_{gender}_{model_name}_{secondary}_box.png")
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--topic_list", nargs="+", default=None)
    parser.add_argument("--probably",action="store_true")
    parser.add_argument("--eval", type=str, default="diversity")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_secondary", type=str, default=None)
    parser.add_argument("--filter", action="store_true")
    
    args = parser.parse_args()
    logger.info(args)

    if args.model_name =="gpt-4":
        model_path = "gpt-4"
    elif args.model_name == "llama2-13b":
        model_path = "meta-llama/Llama-2-13b-hf"
    elif args.model_name == "mistral-7b":
        model_path = "mistralai/Mistral-7B-v0.1"

    if args.topic_list == None:
        args.topic_list = ["favorite_music", "music_instrument", "exercise_routine", "favorite_show_or_movie", "food", "picture_on_the_front_door", "statue_on_the_front_door", "clothing"]
    
    original_data_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}.json"
    shortened_data_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_new_shortened.json"
    cache_dict_path_prefix = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_all_symbols_probability_cache.pkl"
    culture_symbol_path_prefix = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_value_to_culture_mapping.json"

    if args.eval == "diversity":
        diversity_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_diversity_evaluation_count_filtered.json"
        if args.plot:
            # plot_world_map_with_diversity(diversity_save_path, args.model_name, args.topic_list)
            # plot_world_map_with_diversity(diversity_save_path, args.model_name, args.topic_list, gender="male")
            # plot_world_map_with_diversity(diversity_save_path, args.model_name, args.topic_list, gender="female")
            if args.plot_secondary == "nationality":
                secondary_path = f"{args.home_dir}/dataset_search/nationality_count_documents.pkl"
            elif args.plot_secondary == "topic":
                secondary_path = f"{args.home_dir}/dataset_search/nationality_topic_count.pkl"
            else:
                secondary_path = None
            # plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "simpson", args.topic_list, secondary_ax_path=secondary_path, secondary=args.plot_secondary)
            # plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "simpson", args.topic_list, gender="male", secondary_ax_path=secondary_path, secondary=args.plot_secondary)
            # plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "simpson", args.topic_list, gender="female", secondary_ax_path=secondary_path, secondary=args.plot_secondary)
            plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "count", args.topic_list, secondary_ax_path=secondary_path, secondary=args.plot_secondary)
            # plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "count", args.topic_list, gender="male", secondary_ax_path=secondary_path, secondary=args.plot_secondary)
            # plot_bar_chart_with_diversity(args.home_dir,diversity_save_path, args.model_name, "count", args.topic_list, gender="female", secondary_ax_path=secondary_path, secondary=args.plot_secondary)
        else:
            if args.model_name == "gpt-4":
                symbols_path = f"{args.home_dir}/probable_data/categories_nationality_100_gpt-4_prob=True_all_symbols_prefixed.json"
                eval_gpt4_diversity(args.home_dir, shortened_data_path, symbols_path, diversity_save_path, args.topic_list)
                eval_gpt4_diversity(args.home_dir, shortened_data_path, symbols_path, diversity_save_path, args.topic_list, gender="male")
                eval_gpt4_diversity(args.home_dir, shortened_data_path, symbols_path, diversity_save_path, args.topic_list, gender="female")
            else:
                eval_diversity(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, diversity_save_path, args.topic_list)
                eval_diversity(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, diversity_save_path, args.topic_list, gender="male")
                eval_diversity(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, diversity_save_path, args.topic_list, gender="female")

    elif args.eval == "skewness":
        skewness_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_skewness_evaluation_count.json"
        if args.plot:
            plot_world_map_with_skewness(skewness_save_path, args.model_name, args.topic_list)
            plot_world_map_with_skewness(skewness_save_path, args.model_name, args.topic_list, gender="male")
            plot_world_map_with_skewness(skewness_save_path, args.model_name, args.topic_list, gender="female")
        else:
            eval_skewness(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, skewness_save_path, args.topic_list)
            eval_skewness(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, skewness_save_path, args.topic_list, gender="male")
            eval_skewness(args.home_dir, shortened_data_path, cache_dict_path_prefix, culture_symbol_path_prefix, skewness_save_path, args.topic_list, gender="female")
    
    elif args.eval == "markedness":
        markedness_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_markedness_evaluation.json"
        if args.plot:
            # uncomment for plotting world map
            # plot_world_map_with_markedness(markedness_save_path, args.model_name, args.topic_list, aggregate=True)
            # plot_world_map_with_markedness(markedness_save_path, args.model_name, args.topic_list, gender="male")
            # plot_world_map_with_markedness(markedness_save_path, args.model_name, args.topic_list, gender="female")
            # if args.plot_secondary == "nationality":
            #     secondary_path = f"{args.home_dir}/dataset_search/nationality_count_documents.pkl"
            # elif args.plot_secondary == "topic":
            #     secondary_path = f"{args.home_dir}dataset_search/nationality_topic_count.pkl"
            # else:
            #     secondary_path = None

            # uncomment for plotting bar chart
            # plot_bar_chart_with_markedness(markedness_save_path, args.model_name, "markedness", args.topic_list, secondary_ax_path=secondary_path, secondary=args.plot_secondary)

            # uncomment for plotting continentwise markedness
            model_names = ["gpt-4", "llama2-13b", "mistral-7b"]
            markedness_paths = [f"{args.home_dir}/probable_data/categories_nationality_100_{model_name}_prob={args.probably}_markedness_evaluation.json" for model_name in model_names]
            plot_continentwise_markedness(args.home_dir, markedness_paths, model_names, args.topic_list)

            # uncomment for plotting aggregated markedness
            # plot_bar_chart_with_markedness_by_topic(markedness_paths, model_names, args.topic_list)
            
        else:
            eval_markedness(args.home_dir, original_data_path, markedness_save_path, args.topic_list)
    
    elif args.eval == "culture_agnostic":
        culture_agnostic_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_culture_agnostic_overlap_evaluation.json"
        filtered_culture_symbol_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_value_to_culture_mapping_filtered.json"

        if args.filter:
            remove_culture_symbols_generated_in_agnostic_prompts(culture_symbol_path_prefix, culture_agnostic_save_path, filtered_culture_symbol_path)
        elif args.plot:
            if args.plot_secondary == "nationality":
                secondary_path = f"{args.home_dir}/dataset_search/nationality_count_documents.pkl"
            elif args.plot_secondary == "topic":
                secondary_path = f"{args.home_dir}/dataset_search/nationality_topic_count.pkl"
            else:
                secondary_path = None
            plot_culture_symbols_generated_in_agnostic_prompts(args.home_dir, culture_agnostic_save_path, culture_symbol_path_prefix, args.model_name, secondary_ax_path=secondary_path, secondary=args.plot_secondary)
        else:
            if args.model_name == "gpt-4":
                eval_gpt4_culture_symbol_presence_in_culture_neutral_prompt(args.home_dir, shortened_data_path, culture_symbol_path_prefix, culture_agnostic_save_path, args.topic_list, gender="")
            else:
                eval_culture_symbol_presence_in_culture_neutral_prompt(args.home_dir, shortened_data_path, culture_symbol_path_prefix, culture_agnostic_save_path, args.topic_list, gender="")
    
    elif args.eval == "correlation":
        # uncomment for diversity correlation with dataset topic frequency
        diversity_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_diversity_evaluation_count_filtered.json"
        calculate_diversity_correlation_with_training_data(diversity_save_path, eval_type="count", topic_list=None, gender="")

        # uncomment for markedness correlation with dataset topic frequency
        markedness_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_markedness_evaluation.json"
        calculate_markedness_correlation_with_training_data(markedness_save_path, eval_type="vocab_mark", topic_list=None, gender="")