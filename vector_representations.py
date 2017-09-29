from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from matplotlib import pyplot as plt
from scipy import vstack
from nltk.stem.porter import PorterStemmer
import configparser
from self_defined_exceptions import NoArgumentException
from self_defined_exceptions import InvalidArgumentException

# divide 20news data into different groups
def group_data(twenty_train):
    group_map = {}
    for index in range(len(twenty_train.data)):
        if twenty_train.target[index] in group_map:
            group_map[twenty_train.target[index]].append(index)
        else:
            group_map[twenty_train.target[index]] = [index]

    # group_map stores index of group and index of documents in certain group
    return group_map


def construct_matrix_and_group(vectorizer):
    twenty_train = fetch_20newsgroups(subset="all", shuffle=True)
    group_matrix = vectorizer.fit_transform(twenty_train.data)
    groups = group_data(twenty_train)
    return group_matrix, groups


def calculate_similarity_in_one_group(matrix, groups_map):
    num_of_groups = len(groups_map.keys())
    similarity_matrix = np.ones((num_of_groups, num_of_groups))
    for first in range(num_of_groups):
        for second in range(first, num_of_groups):
            average_similarity = cosine_similarity(matrix[groups_map[first]], matrix[groups_map[second]]).mean()
            similarity_matrix[first][second] = average_similarity
            similarity_matrix[second][first] = average_similarity
    return similarity_matrix


def divide_data_in_group(groups_map, number):
    representative_group = {key: value[:number] for key, value in groups_map.items()}
    query_group = {key: value[number:] for key, value in groups_map.items()}
    return representative_group, query_group


def calculate_similarity_between_groups(matrix, groups_map):
    num_of_groups = len(groups_map.keys())
    representative_group, query_group = divide_data_in_group(groups_map, 100)
    similarity_matrix = np.ones((len(query_group.keys()), len(representative_group.keys())))
    for first in range(num_of_groups):
        for second in range(num_of_groups):
            average_similarity = cosine_similarity(matrix[query_group[first]],
                                                   matrix[representative_group[second]]).mean()
            similarity_matrix[first][second] = average_similarity
    return similarity_matrix


def calculate_precision_and_recall(query_group, matrix, threshold):
    indices_list = [ele for indices in query_group.values() for ele in indices]
    query_set = matrix[indices_list]
    precision_list = []
    recall_list = []
    for key, value in query_group.items():
        query_matrix = matrix[value]
        num_true_positive = calculate_num_qualified(threshold, query_matrix)
        num_all_positive = calculate_num_qualified(threshold, query_matrix, query_set)
        precision = (num_true_positive / num_all_positive).mean()
        recall = (num_true_positive / len(value)).mean()
        precision_list.append(precision)
        recall_list.append(recall)

    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    return average_precision, average_recall


def calculate_num_qualified(threshold, first, second=None):
    similarity_matrix = cosine_similarity(first, second) if second is not None else cosine_similarity(first)
    return np.sum(similarity_matrix > threshold, axis=1)


def create_analyzer(old_analyzer, stemmer):
    def steamed_words(document):
        return [stemmer.stem(word) for word in old_analyzer(document)]
    return steamed_words


def create_tokenizer(tokenizer, stemmer):
    def tokenize(document):
        tokens = tokenizer(document)
        return [stemmer.stem(token) for token in tokens]
    return tokenize


def load_npy_file(file_path):
    return np.load(file_path)


def test():
    vectorizer_1 = CountVectorizer()

    tokenizer = vectorizer_1.build_tokenizer()
    stemmer = PorterStemmer()
    matrix, groups_map = construct_matrix_and_group(vectorizer_1)
    _, query_group = divide_data_in_group(groups_map, 100)
    for x in np.arange(0.1, 1.0, 0.1):
        print(calculate_precision_and_recall(query_group, matrix, x))


def read_configs(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    config_dict = {key: string_to_attributes(value) for key, value in config["arguments"].items()}
    return config_dict


def string_to_attributes(input_string):
    if input_string.lower() in {"yes", "true"}:
        return True
    if input_string.lower() in {"no", "false"}:
        return False

    return input_string


def verify_configs(config_dict):
    if not config_dict["vectorizer"]:
        raise NoArgumentException("Please specify the type of Vectorizer")
    if config_dict["vectorizer"].lower() not in ["count", "tf-idf", "none"]:
        raise InvalidArgumentException("The Vectorizer type you specified in invalid")

    if config_dict["read_matrix_from_file"] and not config_dict["matrix_file_path"]:
        raise NoArgumentException("Please specify the path of matrix file")

    if config_dict["vectorizer"].lower() == "none" and not config_dict["read_matrix_from_file"]:
        raise InvalidArgumentException("Please specify the matrix file path")


def build_vectorizer(vectorizer_type, use_stemmer):
    tokenizer = create_tokenizer(CountVectorizer.build_tokenizer(), PorterStemmer()) if use_stemmer else None
    if vectorizer_type == "count":
        return CountVectorizer(tokenizer=tokenizer, stop_words="english")

    if vectorizer_type == "tf-idf":
        return TfidfVectorizer(tokenizer=tokenizer, stop_words="english")

    return None


def do_experiment(config_path):
    config_dict = read_configs(config_path)
    verify_configs(config_dict)
    vecotrizer = build_vectorizer(config_dict["vectorizer"], config_dict["use_porter_stemmer"])
    twenty_train = fetch_20newsgroups(subset="all", shuffle=True)
    group_map = group_data(twenty_train)

    if config_dict["read_matrix_from_file"]:
        matrix = np.load(config_dict["matrix_file_path"])
        if config_dict["save_matrix"]:
            np.save(config_dict["save_matrix_path"])

    else:
        matrix = vecotrizer.fit_transform(twenty_train.data)

    if config_dict["calculate_similarity_one_group"]:
        one_group_matrix = calculate_similarity_in_one_group(matrix, group_map)
        if config_dict["visualize"]:
            print(one_group_matrix)

    if config_dict["calculate_similarity_different_group"]:
        two_group_matrix = calculate_similarity_between_groups(matrix, group_map)
        if config_dict["visualize"]:
            print(two_group_matrix)

    if config_dict["calculate_precision_and_recall"]:
        query_group = {key: value[100:] for key, value in group_map.items()}
        result_list = (calculate_precision_and_recall(query_group, matrix, i) for i in np.arange(0.1, 1.0, 0.1))
        for ele in result_list:
            print(ele)

if __name__ == '__main__':
    config_path = "/Users/zxj/PycharmProjects/cs535/task_parameters.ini"
    do_experiment(config_path)