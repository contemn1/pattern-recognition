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
    print(matrix.shape)
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
    origin = CountVectorizer() if vectorizer_type == "count" else TfidfVectorizer()
    if vectorizer_type == "count":
        if use_stemmer:
            tokenizer = create_tokenizer(origin.build_tokenizer(), PorterStemmer())
            return CountVectorizer(tokenizer=tokenizer)
        else:
            return CountVectorizer()

    if vectorizer_type == "tf-idf":
        if use_stemmer:
            tokenizer = create_tokenizer(origin.build_tokenizer(), PorterStemmer())
            return TfidfVectorizer(tokenizer=tokenizer)
        else:
            return TfidfVectorizer()

    return None


def graw_graph(recall, precision):
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
        np.mean(precision)))
    plt.show()


def print_max_and_min(matrix):
    first = matrix.argmax()
    second = matrix.argmin()
    print("index of max in matrix is {0}".format(np.unravel_index(first, matrix.shape)))
    print("index of min in matrix is {0}".format(np.unravel_index(second, matrix.shape)))


def do_experiment(config_path):
    config_dict = read_configs(config_path)
    verify_configs(config_dict)
    vecotrizer = build_vectorizer(config_dict["vectorizer"], config_dict["use_porter_stemmer"])

    if config_dict["read_matrix_from_file"]:
        twenty_train = fetch_20newsgroups(subset="all", shuffle=True)
        matrix = np.load(config_dict["matrix_file_path"])
        group_map = group_data(twenty_train)

    else:
        matrix, group_map = construct_matrix_and_group(vecotrizer)
        if config_dict["save_matrix"]:
            np.save(config_dict["save_matrix_path"])

    print("shape of matrix is {0}".format(matrix.shape))

    sparseness = np.sum(matrix > 0, axis=1).mean()
    print("sparseness of matrix is {0}".format(sparseness))

    if config_dict["calculate_similarity_one_group"]:
        one_group_matrix = calculate_similarity_in_one_group(matrix, group_map)

        print_max_and_min(one_group_matrix)
        if config_dict["visualize"]:
            output_format = [["{0:.2f}".format(ele) for ele in row] for row in one_group_matrix]
            output_format = [" ".join(row) for row in output_format]
            for ele in output_format:
                print(ele)

            plt.matshow(one_group_matrix)
            plt.show()


    if config_dict["calculate_similarity_different_group"]:
        two_group_matrix = calculate_similarity_between_groups(matrix, group_map)
        print_max_and_min(two_group_matrix)
        if config_dict["visualize"]:
            output_format = [["{0:.2f}".format(ele) for ele in row] for row in two_group_matrix]
            output_format = [" ".join(row) for row in output_format]
            for ele in output_format:
                print(ele)

            plt.matshow(two_group_matrix)
            plt.show()


    if config_dict["calculate_precision_and_recall"]:
        _, query_group = divide_data_in_group(group_map, 100)
        precision_list = []
        recall_list = []
        for x in np.arange(0.0, 1.0, 0.1):
            precision, recall = calculate_precision_and_recall(query_group, matrix, x)
            precision_list.append(precision)
            recall_list.append(recall)
        graw_graph(recall_list, precision_list)


def check_news_group_name():
    twenty_train = fetch_20newsgroups(subset="all", shuffle=True)
    print(twenty_train.target_names)
    names_list = list(twenty_train.target_names)
    print(names_list[5])
    print(names_list[6])
    print(names_list[15])


if __name__ == '__main__':
    check_news_group_name()