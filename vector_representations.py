from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from matplotlib import pyplot as plt


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


def calculate_similarity_between_groups(matrix, groups_map):
    num_of_groups = len(groups_map.keys())
    representative_group = {key: value[:100] for key, value in groups_map.items()}
    query_group = {key: value[100:] for key, value in groups_map.items()}
    similarity_matrix = np.ones((len(query_group.keys()), len(representative_group.keys())))
    for first in range(num_of_groups):
        for second in range(num_of_groups):
            average_similarity = cosine_similarity(matrix[groups_map[first]], matrix[groups_map[second]]).mean()
            similarity_matrix[first][second] = average_similarity
    return similarity_matrix


if __name__ == '__main__':
    vectorizer_1 = CountVectorizer()
    matrix, groups_map = construct_matrix_and_group(vectorizer_1)
    similarity_matrix = calculate_similarity_in_one_group(matrix, groups_map)
    print(similarity_matrix.max())