import pymongo
from redis import StrictRedis 

def connect_to_mongo_and_reference_test_collections():
    mongo_client = pymongo.MongoClient(host='54.213.90.224',port=27016)
    database_ref = mongo_client.test_db
    collection_ref = database_ref.test_coll
    processed_coll_ref = database_ref.processed_test
    return collection_ref, processed_coll_ref

def pull_one_unprocessed_test_item(collection_ref):
    return next(collection_ref.find({'processed' : {'$exists': False}}))

def process_one_item(test_item, processed_coll_ref):
    
    this_processed_bar = test_item
    this_processed_bar['class'] = this_processed_bar['bar'] > 0.5
    processed_coll_ref.insert_one(this_processed_bar)
    
def update_test_item(collection_ref, test_item):
    collection_ref.update_one({'_id':test_item['_id']}, {'$set':{'processed':True}})
    
def pull_and_process_test_item():
    collection_ref, processed_coll_ref = connect_to_mongo_and_reference_test_collections()
    
    this_bar = pull_one_unprocessed_test_item(collection_ref)
    
    try:
        process_one_item(this_bar, processed_coll_ref)
    except pymongo.errors.DuplicateKeyError:
        pass
        
    
    update_test_item(collection_ref, this_bar)
    
def remove_punctuation(document):
    punctuation = ["'", '"', '`', '~', '!', '@', '#', '$', '%',
                   '^', '&', '*', '(', ')', '-', '_', '+', '=',
                   '{', '}', '[', ']', '|', '\\', ':', ';', ',',
                   '<', '>', '.', '/', '?']
    
    for symbol in punctuation:
        document = document.replace(symbol, '')
    return document

def mapper(document, word_list):
    document = remove_punctuation(document)
    words = document.lower().split()
    for word in words:
        token = 1
        push_word(word, token)
        set_add_word(word_list, word)

def reducer(word, count_list):
    count = 0
    token = pop_word(word)
    while token:
        token = int(token.decode())
        count += token
        token = pop_word(word)

    push_count(count_list, word, count)

def toggle_hold():
    hold = check_hold()
    if hold: 
        set_hold(False)
    else:
        set_hold(True)
    
def check_hold():
    redis_connection = StrictRedis('this_redis')
    return redis_connection.get('hold') == b'True'
    
def set_hold(value):
    redis_connection = StrictRedis('this_redis')
    redis_connection.set('hold', value)
    
def pop_word(word):
    redis_connection = StrictRedis('this_redis')
    return redis_connection.lpop(word)

def push_count(count_list, word, count):

    redis_connection = StrictRedis('this_redis')
    redis_connection.rpush(count_list, (word, count))
    
def push_word(word, token):
    redis_connection = StrictRedis('this_redis')
    redis_connection.rpush(word, token)
    

def set_add_word(word_list, word):
    redis_connection = StrictRedis('this_redis')
    redis_connection.sadd(word_list, word)