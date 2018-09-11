import csv
import time

import numpy as np
from numpy.linalg import norm


average_rating = {}


def read_file():
    name = 'Dataset/ratings.csv'
    temp = []
    global dick_item, dick_user, total_item, total_user
    with open(name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            temp.append([int(row[0]), int(row[1]), float(row[2])])
    dataset = np.asarray(temp)
    np.random.shuffle(dataset)
    return dataset


def find_similarity(matrix):
    similarity = {}
    user_list = list(matrix.keys())
    for i in range(len(user_list)):
        key1 = user_list[i]
        similarity[key1] = {}
        for j in range(len(user_list)):
            key2 = user_list[j]
            if j < i:
                similarity[key1][key2] = similarity[key2][key1]
            else:
                similarity[key1][key2] = cosine_similarity(key1, key2, matrix)
    return similarity


def cosine_similarity(user1, user2, matrix):
    sim = 0
    ord1 = 0
    ord2 = 0
    for key in matrix[user1]:
        if key in matrix[user2]:
            sim += (matrix[user1][key]*matrix[user2][key])
            ord1 += matrix[user1][key]*matrix[user1][key]
            ord2 += matrix[user2][key]*matrix[user2][key]
    denom = np.sqrt(ord1) * np.sqrt(ord2)
    if denom == 0:
        sim = 0
    else:
        sim /= denom
    return sim


def get_matrix(train):
    global average_rating
    average_rating = {}
    count = {}
    user_vectors = {}
    for values in train:
        user = int(values[0])
        movie = int(values[1])
        if user not in user_vectors:
            user_vectors[user] = {}
            average_rating[user] = 0
            count[user] = 0
        # check if we need to use the 0 values
        user_vectors[user][movie] = values[2]
        average_rating[user] += values[2]
        count[user] += 1
    for key in average_rating:
        average_rating[key] /= count[key]
    return user_vectors


def model(test, train, n_count):
    matrix = get_matrix(train)
    print("Matrix created")
    # print(average_rating)
    similarity = find_similarity(matrix) #check if error comes here later
    print("Similarity calculated")
    error = get_mean_error(test, matrix, similarity, n_count)
    # print("Average error = ", error)
    return error


def predict(user, movie, similarity, matrix, n_count ):
    global average_rating
    rating = 0
    k = n_count
    neighbours = []
    for key in similarity[user]:
        neighbours.append( (similarity[user][key], key) )
    neighbours.sort(reverse = True)
    neighbours = neighbours[:k]

    sum_sim = 0
    for val1 in neighbours:
        val = val1[1]
        if movie in matrix[val]:
            rating += (similarity[user][val] * (matrix[val][movie] - average_rating[val] ))
            sum_sim += similarity[user][val]

    if sum_sim == 0:
        return average_rating[user]
    else:
        rating /= sum_sim
        rating += average_rating[user]
        return rating


def get_mean_error(test, matrix, similarity, n_count):
    error = 0
    for values in test:
        rating = predict(values[0], values[1], similarity, matrix, n_count )
        # print('Prediction ', rating, 'vs Actual ', values[2])
        error += abs( values[2] - rating )
    error /= len(test)
    return error


def k_fold(dataset, n_count):
    error = 0.0
    k = int(len(dataset) / 5)

    start = time.time()
    print("Iteration 1")
    test = dataset[:k]
    train = dataset[k:]
    temp = model(test, train, n_count)
    error += temp
    print(temp)

    print("Iteration 2")
    test = dataset[k:2 * k]
    train = np.concatenate((dataset[:k], dataset[2 * k:]))
    temp = model(test, train, n_count)
    error += temp
    print(temp)

    print("Iteration 3")
    test = dataset[2 * k:3 * k]
    train = np.concatenate((dataset[:2 * k], dataset[3 * k:]))
    temp = model(test, train, n_count)
    error += temp
    print(temp)

    print("Iteration 4")
    test = dataset[3 * k:4 * k]
    train = np.concatenate((dataset[:3 * k], dataset[4 * k:]))
    temp = model(test, train, n_count)
    error += temp
    print(temp)

    print("Iteration 5")
    test = dataset[4 * k:]
    train = dataset[:4 * k]
    temp = model(test, train, n_count)
    error += temp
    print(temp)

    end = time.time()
    print("Time taken = ", end - start)
    print(error / 5)

    return error / 5



def main():
    dataset = read_file()
    error = []
    n_count = [10, 20, 30, 40, 50]
    for n in n_count:
        error.append(k_fold(dataset, n))

    print(error)



main()