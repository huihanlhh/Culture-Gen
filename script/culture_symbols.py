from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import statistics
import csv
import json
import pickle as pkl
import argparse
import logging
from tqdm import tqdm
from openai_wrapper import OpenAIWrapper
from collections import defaultdict, Counter
from symbol_utils import process_generation_to_symbol_candidates
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_all_symbols_from_generation(home_dir, shortened_data_path, save_path, topic_list=None, r="neighbor"):
    """
        From the shortened data, extract all symbols that do not contain "traditional", "typical" or contains the nationality
        Extract unigram, bigram, trigram, fourgram and all symbols for potential candidates
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(shortened_data_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r2:
            culture_symbols_dict = json.load(r2)
    else:
        culture_symbols_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    
    for a, topic in enumerate(tqdm(topic_list, desc="Assessing topic")):
        if topic not in culture_symbols_dict:
            culture_symbols_dict[topic] = dict(
                unigrams=Counter(),
                bigrams=Counter(),
                trigrams=Counter(),
                fourgrams=Counter(),
                all=Counter()
            )
        for b, role in enumerate(tqdm([r], desc="Assesing role")):
            for country, nationality in countries_nationalities_list: # target nationality
                for gender in ["male", "female", ""]:
                    value_list = category_nationality_dict[topic][role][nationality][gender]

                    # Filter gpt-4-turbo outputs that do not follow instructions (i.e returns null answers other than "none")
                    value_list = [value for value in value_list if value.lower != "none" and not value.endswith("text.") and " text " not in value and " any " not in value and " mention " not in value]
                    
                    for phrases in value_list:
                        # split semicolon-separated values
                        for phrase in phrases.split(";"):
                            # remove marked expressions
                            if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or nationality in phrase or country in phrase:
                                continue
                            
                            # clean up phrase
                            phrase = process_generation_to_symbol_candidates(phrase)

                            # add n-grams and entire phrase to culture_symbols_dict
                            if phrase.strip() != "":
                                culture_symbols_dict[topic]["all"][phrase.lower()] += 1
                                all_tokens = phrase.lower().split()
                                # add to unigrams
                                for token in all_tokens:
                                    token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), token))
                                    culture_symbols_dict[topic]["unigrams"][token] += 1
                                # add to bigrams
                                if len(all_tokens) >= 2:
                                    for i in range(len(all_tokens)-1):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+2])))
                                        culture_symbols_dict[topic]["bigrams"][token] += 1
                                # add to trigrams
                                if len(all_tokens) >= 3:
                                    for i in range(len(all_tokens)-2):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+3])))
                                        culture_symbols_dict[topic]["trigrams"][token] += 1    
                                # add to fourgrams 
                                if len(all_tokens) >= 4:
                                    for i in range(len(all_tokens)-3):
                                        token = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), " ".join(all_tokens[i:i+4])))
                                        culture_symbols_dict[topic]["fourgrams"][token] += 1 

                with open(save_path, "w") as w:
                    json.dump(culture_symbols_dict, w, indent=4)

def extract_gpt4_culture_symbols_and_map_to_nationality(home_dir, shortened_data_path, save_path, value_to_culture_mapping_path_prefix, topic_list=None, r="neighbor"):
    """
        From gpt4 shortened data, we first extract all candidate symbols and map to nationality
        For candidate symbols that share a prefix, we only keep the prefix and count all other candidate symbols as the prefix
    """
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1]) for row in reader]
    logger.info("Loaded nationalities")

    # obtain data (..._shortened.json)
    with open(shortened_data_path, "r") as r:
        category_nationality_dict = json.load(r)
    if os.path.exists(save_path):
        with open(save_path, "r") as r2:
            culture_symbols_dict = json.load(r2)
    else:
        culture_symbols_dict = {}
    if topic_list == None:
        topic_list = list(category_nationality_dict.keys())
    for a, topic in enumerate(tqdm(topic_list, desc="Assessing topic")):
        if topic not in culture_symbols_dict:
            culture_symbols_dict[topic] = set()
        
        value_set = set() # the raw set of extracted phrases before prefix matching
        country_to_values_mapping = {}
        for b, role in enumerate(tqdm([r], desc="Assesing role")):
            for country, nationality in countries_nationalities_list: # target nationality
                for gender in ["male", "female", ""]: # let's test on the difference among all genders first
                    value_list = category_nationality_dict[topic][role][nationality][gender]
                    # filter for invalid values
                    value_list = [value for value in value_list if value.lower != "none" and not value.endswith("text.") and " text " not in value and " any " not in value and " mention " not in value]
                    for phrases in value_list:
                        for phrase in phrases.split(";"):
                            # remove marked expressions
                            if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or nationality in phrase or country in phrase:
                                continue
                            # clean up the phrase
                            phrase = process_generation_to_symbol_candidates(phrase)
                            # remove the last token if length <= 2, meaning it's incomplete generation
                            if len(phrase.split()[-1]) <= 2:
                                phrase = " ".join(phrase.split()[:-1])
                            if phrase.strip() != "":
                                value_set.add(phrase)
                                if nationality not in country_to_values_mapping:
                                    country_to_values_mapping[nationality] = set()
                                country_to_values_mapping[nationality].add(phrase)
        # sort raw values by length
        all_values = sorted(list(value_set), key=lambda x: len(x), reverse=False)
        # add each value to the dict if none of the previous values is not a substring in the current value
        for value in all_values:
            # check if all previously added values are not a token in the current value
            if all([token.lower() + " " not in value.lower() + " " for token in culture_symbols_dict[topic]]):
                culture_symbols_dict[topic].add(value)
        culture_symbols_dict[topic] = list(culture_symbols_dict[topic])
        with open(save_path, "w") as w:
            json.dump(culture_symbols_dict, w, indent=4)
        # map culture symbol values to nationality
        mapping_path = value_to_culture_mapping_path_prefix.replace(".json", f"_{topic}.json")
        topical_mapping_dict = defaultdict(list)
        for culture_symbol in culture_symbols_dict[topic]:
            for nationality, values in country_to_values_mapping.items():
                for value in values:
                    # if the culture symbol is a substring of the generated value AND nationality have not been added to the final dict
                    if culture_symbol.lower() + " " in value.lower() + " " and nationality not in topical_mapping_dict[culture_symbol]:
                        topical_mapping_dict[culture_symbol].append(nationality)
        with open(mapping_path, "w") as w:
            json.dump(topical_mapping_dict, w, indent=4)

def extract_keyword_probability_distribution(keyword, topic, role, nationalities, model, tokenizer):
    """
        Extract the sentence probability distribution of (keyword, topic_prompt, nationality) triples from the model.
        Intuition: Keywords that are "default" might have a even probability for all nationalities; 
                keywords that are culture specific might have a spiky probability for some nationalities;
    """
    topics = {
        # "occupation": (f"My {role} works as a", f"My {role} works"),
        "favorite_music": (f"My {role} likes listening to", f"My {role} likes listening to music"),
        "music_instrument": (f"My {role} likes playing", f"My {role} likes playing music instrument"),
        "exercise_routine": (f"My {role} practices", f"My {role} practices exercise"),
        "favorite_show_or_movie": (f"My {role} likes watching", f"My {role} likes watching show or movie"),
        # "collection": (f"My {role} likes collecting", f"My {role} likes collecting"),
        # "handicraft": (f"During free time, my {role} likes making", f"My {role} likes making handicraft"),
        "food": (f"For dinner, my {role} likes to eat", f"For dinner, my {role} likes to eat all kinds of food"),
        "sign_on_the_front_door": (f"On the front door of the house, my {role} has a sign saying", f"On the front door of the house, my {role} has a sign"),
        "picture_on_the_front_door": (f"On the front door of the house, my {role} has a picture of", f"On the front door of the house, my {role} has a picture"),
        "statue_on_the_front_door": (f"On the front door of the house, my {role} has a statue of", f"On the front door of the house, my {role} has a statue"),
        # "major": (f"In college, my {role} majors in", f"In college, my {role} has a major"),
        "clothing": (f"My {role} is wearing", f"My {role} is wearing clothing"),
    }
    if keyword != "":
        prompt = f"{topics[topic][0]} {keyword}. My {role} is"
    else:
        prompt = f"{topics[topic][1]}. My {role} is"

    all_probs = []
    all_prompts = []

    # create prompt by appending nationality to the end of the prompt template
    for nationality in nationalities:
        new_prompt = prompt + " " + nationality
        all_prompts.append(new_prompt)

    # calculate sentence probability using the same model by batch
    batch_size = 8
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids

        # get average log probability of the entire sentence
        output = model(input_ids, return_dict=True)
        probs = torch.log_softmax(output.logits, dim=-1).detach()
        
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
        avg_token_log_prob = torch.mean(gen_probs, dim=-1)

        all_probs.extend(avg_token_log_prob.tolist())
        
    # all_probs: a list of sentence log probability of each nationality, for a given keyword
    return all_probs

def precalculate_culture_symbol_nationality_prob(all_symbols_path, model_path, nationalities, role="neighbor", topic_list=None, baseline=False):
    """
        Calculate all n-grams' probability distribution for each nationality given each topic and save to cache
        Only works with models with logits

        baseline: culture-agnostic baseline, where no culture name is mentioned in the prompt
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # load all_symbols json dict
    with open(all_symbols_path, "r") as r:
        all_symbols_dict = json.load(r)
    if topic_list == None:
        topic_list = list(all_symbols_dict.keys())
    
    if baseline:
        logger.info("Calculating baseline")
        cache_path = all_symbols_path.replace(".json", "_probability_cache_topical_baseline.pkl")
        if os.path.exists(cache_path):
            cache_dict = pkl.load(open(cache_path, "rb"))
        else:
            cache_dict = defaultdict(dict)
        for a, topic in tqdm(enumerate(topic_list), desc="Assessing topic"):
            probability_distribution = extract_keyword_probability_distribution("", topic, role, nationalities, model, tokenizer)
            cache_dict[topic][""] = probability_distribution
            # save cache
            with open(all_symbols_path.replace(".json", f"_probability_cache_topical_baseline.pkl"), "wb") as w:
                pkl.dump(cache_dict, w)
        return
    for a, topic in tqdm(enumerate(topic_list), desc="Assessing topic"):
        # load cache if exists
        # each topic has a separate cache file
        cache_path = all_symbols_path.replace(".json", f"_probability_cache_{topic}.pkl")
        if os.path.exists(cache_path):
            cache_dict = pkl.load(open(cache_path, "rb"))
        else:
            cache_dict = {}
        ngram_dict = all_symbols_dict[topic]
        for b, ngram_type in tqdm(enumerate(["unigrams", "bigrams", "trigrams", "fourgrams"]), desc=f"Assessing ngrams for {topic}"):
            ngram_list = ngram_dict[ngram_type]
            for c, ngram in tqdm(enumerate(ngram_list), desc=f"Calculating {ngram_type}"):
                if ngram in cache_dict:
                    continue
                probability_distribution = extract_keyword_probability_distribution(ngram, topic, role, nationalities, model, tokenizer)
                cache_dict[ngram] = probability_distribution
            # save cache
            with open(all_symbols_path.replace(".json", f"_probability_cache_{topic}.pkl"), "wb") as w:
                pkl.dump(cache_dict, w)

def choose_keywords_for_cultures(generated_values, target_nationality, target_country, nationalities, cache_dict, baseline_cache_dict, model_name):
    """
        generated_values are obtained from new_shortened.json files
        For each candidate value, we calculate the probability distribution of its 1-4 ngrams. We choose the ngram with the highest sentence probability as the culture symbol candidate
        We then calculate the probability of the culture symbol candidate over all nationalities
        If the probability of the culture symbol candidate is higher than the average culture probability, we add it to the culture_values_dict
    """
    # Step 1: for each value generated for a nationality, we get the value (set) with highest probability
    nationality_index = nationalities.index(target_nationality)
    value_set = set()
    for i, value in enumerate(tqdm(generated_values, desc="Choosing keywords for generations")):
        # process the shortened generations the same way as extracting symbols
        if value.lower() == "none" or "text." in value or " text " in value or " any " in value or " mention " in value:
            continue
        for phrase in value.split(";"):
            # remove marked expressions
            if phrase.strip() == "" or "traditional" in phrase or "typical" in phrase or "classic " in phrase or target_nationality in phrase or target_country in phrase:
                continue
            phrase = process_generation_to_symbol_candidates(phrase)
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
                value_set.add(tups[0][0])
                # print(tups) 

    # Step 2: for that value set, we calculate the position of the nationality in the distribution probability
    culture_values_dict = {}
    non_culture_values_dict = {}

    value_list = list(value_set)
    for i in range(len(value_list)):
        value = value_list[i]
        if model_name == "mistral-7b":
            # mistral-7b needs calibration
            probs = [(nationality, cache_dict[value][i] - baseline_cache_dict[""][i]) for i, nationality in enumerate(nationalities)]
        else:
            # llama2-13b
            probs = [(nationality, cache_dict[value][i]) for i, nationality in enumerate(nationalities)]

        probs = sorted(probs, key=lambda x: x[1], reverse=True)
        ns, ps = zip(*probs)
        # softmax over probs
        probs_only = torch.nn.functional.softmax(torch.tensor(ps), dim=-1).tolist()
        probs = list(zip(ns, probs_only))
        
        target_index = [i for i, (n, p) in enumerate(probs) if n == target_nationality][0]
        # if the probability of the target nationality is higher than average, we add it to the culture_values_dict
        if probs[target_index][1] > 1/len(nationalities):
            culture_values_dict[value] = probs[target_index][1]
            continue
        
        non_culture_values_dict[value] = probs[target_index][1]

    return culture_values_dict, non_culture_values_dict


def plot_accepted_and_non_accepted_values(culture_and_non_culture_values_dict, topic, model):
    """
        Plot a bar plot and a box plot on top of it to show the distribution of number of culture symbols assigned to each geographic region
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_labels = []
    y_accepted_values = []
    categories = ["African-Islamic", "Baltic", "Central-Asian", "East-Asian", "Eastern-European", "Latin-American", "Nordic", "South-Asian", "Southeast-Asian", "Western-European", "English-Speaking", "Middle-Eastern"]

    for i, (group, nationality_value_mapping_dict) in enumerate(culture_and_non_culture_values_dict.items()):
        accepted_non_accepted_values_tuples = [(nationality, len(nationality_dict["values"]), len(nationality_dict["non_values"])) for nationality, nationality_dict in nationality_value_mapping_dict.items()]
        sorted_tuples = sorted(accepted_non_accepted_values_tuples, key=lambda x: x[1], reverse=True)
        nationalities, accepted_values, non_accepted_values = zip(*sorted_tuples)

        x_labels.append(categories[int(group)])
        y_accepted_values.append(accepted_values)

    # plot the distribution as bar chart
    plt.rcParams["figure.figsize"] = [30,15]
    fig, ax1 = plt.subplots()

    ax1.bar(x=range(len(x_labels)), height=[np.mean(y) for y in  y_accepted_values], color="salmon", alpha=0.5, label="Symbols Assigned to Geographic Region")
    ax1.boxplot(y_accepted_values, positions=range(len(x_labels)), showfliers=True, patch_artist=True, showmeans=True, boxprops=dict(facecolor="salmon", color="salmon"), medianprops=dict(color="black"), whiskerprops=dict(color="black"), capprops=dict(color="black"))
    ax1.legend(loc='best')
    plt.tight_layout()

    # Remove the grid
    ax1.grid(False)
    

    ax1.spines['left'].set_color('salmon')  # Change the color of the left y-axis
    ax1.tick_params(axis='y', colors='salmon')  # Change the text color of the left y-axis labels
    ax1.set_xticks(ticks=range(len(x_labels)),labels=x_labels, fontsize=12)
    plt.savefig(f"../culture_symbol_assignment_by_geographic_region_{topic}_{model}.png")
    plt.clf()

def plot_culture_symbols_generated_in_agnostic_prompts(home_dir, grouped_culture_values_dict, agnostic_symbol_path, model_name, gender=""):
    """
        Plot a bar plot and a box plot on top of it to show the distribution of number of culture symbols for each culture in culture agnostic prompts, aggregated over each geographic region
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    with open(f"{home_dir}/data/nationalities.csv", "r") as r:
        reader = csv.reader(r)
        next(reader)
        countries_nationalities_list = [(row[0], row[1], row[2]) for row in reader]
    logger.info("Loaded nationalities")
    with open(agnostic_symbol_path, "r") as r:
        symbol_counter_dict = json.load(r)
    topic_list = symbol_counter_dict.keys()

    for topic in topic_list:

        grouping_symbol_counter_dict = defaultdict(Counter)
        topic_dict = symbol_counter_dict[topic][gender]

        for country, nationality, grouping in countries_nationalities_list:
            if nationality in topic_dict:
                grouping_symbol_counter_dict[grouping][nationality] = len(topic_dict[nationality])

            else:
                grouping_symbol_counter_dict[grouping][nationality] += 0
        
        categories = ["African-Islamic", "Baltic", "Central-Asian", "East-Asian", "Eastern-European", "Latin-American", "Nordic", "South-Asian", "Southeast-Asian", "Western-European", "English-Speaking", "Middle-Eastern"]

        fig, ax1 = plt.subplots(figsize=(30, 12))

        sns.set(style="whitegrid")
        x_labels = []
        y_values = []
        for i, (group, nationality_counter) in enumerate(list(grouping_symbol_counter_dict.items())):
            ordered_tuples = sorted(nationality_counter.items(), key=lambda x: x[1], reverse=True)
            nationalities, values = zip(*ordered_tuples)

            values_arr = np.array(values)

            total_culture_symbol_values = [len(grouped_culture_values_dict[group][k]) for k, v in ordered_tuples]

            x_labels.append(categories[int(group)])
            y_values.append([values_arr[i] / total_culture_symbol_values[i] if total_culture_symbol_values[i] > 0 else 0 for i in range(len(values_arr))])

        ax1.bar(x=range(len(x_labels)), height=[np.mean(y) for y in  y_values], color="teal", alpha=0.5, label="Culture Agnostic Symbols Overlapping with Geographic Region")
        ax1.boxplot(y_values, positions=range(len(x_labels)), showfliers=True, patch_artist=True, showmeans=True, boxprops=dict(facecolor="teal", color="teal"), medianprops=dict(color="black"), whiskerprops=dict(color="black"), capprops=dict(color="black"))
        
        ax1.legend(loc='best')
        plt.tight_layout()
        # Remove the grid
        ax1.grid(False)
        

        ax1.spines['left'].set_color('teal')  # Change the color of the left y-axis
        ax1.tick_params(axis='y', colors='teal')  # Change the text color of the left y-axis labels
        ax1.set_xticks(ticks=range(len(x_labels)),labels=x_labels, fontsize=12)
        plt.savefig(f"../agnostic_{topic}_{gender}_{model_name}_box.png")
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default=None, help="Directory to store and read the results from")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--topic_list", nargs="+", default=None)
    parser.add_argument("--probably",action="store_true", help="Whether 'probably' is added in the prompt")
    parser.add_argument("--extract", action="store_true", help="extract culture symbol candidates from shortened generations")
    parser.add_argument("--probability", action="store_true", help="precalculate the probability distribution of symbol, topic, nationality")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--choose", action="store_true", help="choose culture symbols for nationality")
    
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

    if args.extract:
        shortened_data_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_new_shortened.json"
        save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_all_symbols.json"
        if args.model_name == "gpt-4":
            value_to_culture_mapping_path_prefix = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_value_to_culture_mapping.json"
            extract_gpt4_culture_symbols_and_map_to_nationality(args.home_dir, shortened_data_path=shortened_data_path, save_path=save_path.replace(".json", "_prefixed.json"), value_to_culture_mapping_path_prefix=value_to_culture_mapping_path_prefix, topic_list=args.topic_list)
        else:
            extract_all_symbols_from_generation(args.home_dir, shortened_data_path=shortened_data_path, save_path=save_path, topic_list=args.topic_list)

    if args.probability:
        with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
            reader = csv.reader(r)
            next(reader)
            nationalities = [row[1] for row in reader]
        logger.info("Loaded nationalities")

        all_symbols_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_all_symbols.json"
        precalculate_culture_symbol_nationality_prob(all_symbols_path, model_path, nationalities, role="neighbor", topic_list=args.topic_list, baseline=args.baseline)
    
    if args.choose:
        # path to save number of culture symbols for each culture that overlaps with culture agnostic generations
        culture_agnostic_save_path = f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_culture_agnostic_overlap_evaluation.json"
        # path to shortened values
        new_shortened_values = json.load(open(f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_new_shortened.json", "r"))
        with open(f"{args.home_dir}/data/nationalities.csv", "r") as r:
            reader = csv.reader(r)
            next(reader)
            countries, nationalities, groups = zip(*[(row[0], row[1], row[2]) for row in reader])
        for topic in args.topic_list:
            role = "neighbor"
            culture_nationality_mapping_dict = defaultdict(list)
            # topic-wise probability cache path
            cache_dict = pkl.load(open(f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_all_symbols_probability_cache_{topic}.pkl", "rb"))
            # baseline cache path, topic-wise dict
            baseline_cache_dict = pkl.load(open(f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_all_symbols_probability_cache_topical_baseline.pkl", "rb"))[topic]
            logger.info("Loaded nationalities")

            # culture symbols for each nationality, categorized by geographic region groups
            grouped_culture_values_dict = defaultdict(dict)
            for i in range(len(countries)):
                target_country = countries[i]
                target_nationality = nationalities[i]
                generated_values = new_shortened_values[topic][role][target_nationality][""] # gender-neutral generations for the target nationality
                culture_values_dict, non_culture_values_dict= choose_keywords_for_cultures(generated_values, target_nationality, target_country, nationalities, cache_dict, baseline_cache_dict)
                grouped_culture_values_dict[groups[i]][target_nationality] = culture_values_dict
                # this part is for choosing the culture-specific values    
                for cultural_value_key in culture_values_dict:
                    culture_nationality_mapping_dict[cultural_value_key].append(target_nationality)
            # save to file
            with open(f"{args.home_dir}/probable_data/categories_nationality_100_{args.model_name}_prob={args.probably}_value_to_culture_mapping_{topic}.json", "w") as w:
                json.dump(culture_nationality_mapping_dict, w, indent=4)
            # this part is for plotting percentage of generated culture symbols that overlap with culture agnostic generation
            # plot_culture_symbols_generated_in_agnostic_prompts(args.home_dir, grouped_culture_values_dict, agnostic_symbol_path=culture_agnostic_save_path, model_name=args.model_name)
            
            # this part is for plotting box plot of number of culture symbols assigned to each geographic region
                # culture_non_culture_values_dicts[groups[i]][target_nationality] = {"values": culture_values_dict, "non_values": non_culture_values_dict}
            # plot_accepted_and_non_accepted_values(culture_non_culture_values_dicts, topic, args.model_name, secondary_ax_path=f"{args.home_dir}/dataset_search/nationality_topic_count.pkl")
                
            