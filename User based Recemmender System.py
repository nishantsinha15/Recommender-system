import csv
import numpy as np
from numpy.linalg import norm

# todo split the matrix - k fold
# todo : function to return user vectors
# done : function which given a user id will find the similarity of all users compared to it
# done : function for coosine similarity
# todo : add features of relevance and variance
# todo : function which will accept a k and return a mean average error
    # todo : given a k and the neighbourhood of a user, predict the ratings for that user

total_user = 0
total_item = 0
dick_user = {}
dick_item = {}


def read_file():
    name = 'Dataset/ratings.csv'
    temp = []
    user = 0
    item = 0
    limit = 0
    global dick_item, dick_user, total_item, total_user
    with open(name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if ++limit > 10:
                break
            temp.append([int(row[0]), int(row[1]), float(row[2])])
            if int(row[0]) not in dick_user:
                dick_user[int(row[0])] = user
                user += 1
            if int(row[1]) not in dick_item:
                dick_item[int(row[1])] = item
                item += 1
    total_item = item
    total_user = user

    dataset = np.asarray(temp)
    return dataset


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def calculate_similarity(user_vector, matrix_train):
    similarity = []
    i = 0
    for user in matrix_train:
        sim_val = cosine_similarity(user_vector, user) #add more features here
        similarity.append((sim_val, i))
        i+=1
    similarity.sort(reverse=True)
    similarity = np.asarray(similarity)
    return similarity


def predict_rating( matrix_train, similarity ):
    prediction = np.zeros([1,len(matrix_train[0]) ])
    div = 0
    for rows in similarity:
        prediction += matrix_train[rows[1]]*rows[0]
        div += rows[0]
    prediction /= div
    return prediction


def get_error( user_vector, matrix_train, k ):
    similarity = calculate_similarity(user_vector, matrix_train)
    similarity = similarity[0:k]
    prediction = predict_rating( matrix_train, similarity )
    error = 0.0
    count = 0
    for i in range(len(user_vector)):
        if user_vector[i] != 0:
            error += abs(user_vector[i] - prediction[0][i])
            count += 1
    error /= count
    return error


def create_matrix(dataset):
    matrix = np.zeros([total_user, total_item])
    for row in dataset:
        matrix[dick_user[row[0]]][dick_item[row[1]]] = row[2]
    return matrix


def main():
    dataset = read_file()
    matrix = create_matrix(dataset)
    print(matrix)


main()
