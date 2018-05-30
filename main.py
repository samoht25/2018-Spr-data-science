from flask import Flask, render_template, request
from redis import Redis
from lib.helper import get_iris_data, matrix_from_redis, plot, reset, sample, train
import pickle

app = Flask(__name__)
redis_connection = Redis('this_redis')
@app.route("/")
def hello():
    return "API is live!"

@app.route("/redis/<key>")
def redis_get(key):
    value = redis_connection.get(key)
    if value:
        value = value.decode()
    return "<h1>Your Value</h1><p>{}: {}</p>".format(key, value)

@app.route("/matrix-elaborate/<key>")
def redis_matrix(key):
    value = redis_connection.get(key)
    if value:
        value = matrix_from_redis(redis_connection, key)
    return "<h1>Your Value</h1><p>{}: {}</p>".format(key, value)

@app.route("/matrix-better/<key>")
def redis_matrix_better(key):
    value = redis_connection.get(key)
    if value:
        value = pickle.loads(value)
    return "<h1>Your Value</h1><p>{}: {}</p>".format(key, value)

@app.route("/iris/<action>", methods=['GET', 'POST'])
@app.route("/iris", methods=['GET'])
def iris(action=None):

    iris_X, iris_y, sample_indices = get_iris_data()
    if action == 'reset':
        reset()
        iris_X, iris_y, sample_indices = get_iris_data()
    if action == 'sample':
        sample()
        iris_X, iris_y, sample_indices = get_iris_data()
    if action == 'train':
        train(iris_X, iris_y, sample_indices)
    results = plot(iris_X, iris_y, sample_indices)

    return render_template('iris.html', results=results, action=action)
