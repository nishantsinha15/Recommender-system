import csv
import random
import time
import numpy as np
from numpy.linalg import norm

average_rating_user = {}
average_rating_item = {}


# done case when item doesn't exist in memory

def get_item_matrix(train):
    global average_rating_item
    average_rating_item = {}
    count = {}
    item_vectors = {}
    for items in train:
        val = int(items[1] )
        user = int(items[0])
        if val not in average_rating_item:
            item_vectors[val] = {}
            average_rating_item[val] = 0
            count[val] = 0
        item_vectors[val][user] = items[2]
        count[val] += 1
        average_rating_item[val] += items[2]
    for key in count:
        average_rating_item[key] /= count[key]
    return item_vectors


def read_file():
    name = 'Dataset/ratings.csv'
    dataset = []
    global dick_item, dick_user, total_item, total_user
    with open(name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset.append([int(row[0]), int(row[1]), float(row[2])])
    # dataset = np.asarray(temp)
    random.shuffle(dataset)
    return dataset


def get_matrix(train):
    global average_rating_user
    average_rating_user = {}
    count = {}
    user_vectors = {}
    for values in train:
        user = int(values[0])
        movie = int(values[1])
        if user not in user_vectors:
            user_vectors[user] = {}
            average_rating_user[user] = 0
            count[user] = 0
        # check if we need to use the 0 values
        user_vectors[user][movie] = values[2]
        average_rating_user[user] += values[2]
        count[user] += 1
    for key in average_rating_user:
        average_rating_user[key] /= count[key]
    return user_vectors


def cosine_similarity(item1, item2, item_matrix ):
    sim = 0
    ord1 = 0
    ord2 = 0
    for key in item_matrix[item1]:
        if key in item_matrix[item2]:
            sim += (item_matrix[item1][key]*item_matrix[item2][key])
            ord1 += item_matrix[item1][key]*item_matrix[item1][key]
            ord2 += item_matrix[item2][key]*item_matrix[item2][key]
    denom = np.sqrt(ord1) * np.sqrt(ord2)
    if denom == 0:
        sim = 0
    else:
        sim /= denom
    return sim


def get_similarity(item_matrix):
    similarity = {}
    item_list = list(average_rating_item.keys())

    point = -1
    for i in range(len(item_list)):
        point += 1
        if point % 100 == 0:
            print("Iteration ", point)
        key1 = item_list[i]
        similarity[key1] = {}
        for j in range(len(item_list)):
            key2 = item_list[j]
            if j < i:
                similarity[key1][key2] = similarity[key2][key1]
            else:
                similarity[key1][key2] = cosine_similarity(key1, key2, item_matrix)
    return similarity


def model(test, train):
    matrix = get_matrix(train)
    print("Matrix created")
    item_matrix = get_item_matrix(train)
    print("Item Matrix created")
    print("Item Count = ", len(average_rating_item.keys()))
    begin = time.time()
    similarity = get_similarity(item_matrix)
    end = time.time()
    print("Similarity found in ",end-begin)
    error = get_mean_error(test, matrix, similarity)
    # print(error)
    return error


def get_mean_error(test, matrix, similarity):
    error = 0
    for values in test:
        rating = predict(values[0], values[1], similarity, matrix )
        temp = abs( values[2] - rating )
        error += temp
    error /= len(test)
    return error


def predict(user, movie, similarity, matrix ):
    rating = 0
    sum_sim = 0
    if movie not in similarity:
        return 3
    for other_movie in matrix[user]:
        if movie in similarity[other_movie]:
            rating += (similarity[other_movie][movie] * ( matrix[user][other_movie] - average_rating_user[user] ))
            sum_sim += similarity[other_movie][movie]

    if sum_sim == 0:
        return average_rating_item[movie]
    else:
        rating /= sum_sim
        rating += average_rating_item[movie]
        return rating


def main():
    dataset = read_file()
    k_fold(dataset)


def k_fold(dataset):
    error = 0.0
    k = int(len(dataset) / 5)

    start = time.time()
    print("Iteration 1")
    test = dataset[:k]
    train = dataset[k:]
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 2")
    test = dataset[k:2 * k]
    train = np.concatenate((dataset[:k], dataset[2 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 3")
    test = dataset[2 * k:3 * k]
    train = np.concatenate((dataset[:2 * k], dataset[3 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 4")
    test = dataset[3 * k:4 * k]
    train = np.concatenate((dataset[:3 * k], dataset[4 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 5")
    test = dataset[4 * k:]
    train = dataset[:4 * k]
    temp = model(test, train)
    error += temp
    print(temp)

    end = time.time()
    print("Time taken = ", end - start)
    print("Final Error = ", error / 5)
    return error / 5


main()


