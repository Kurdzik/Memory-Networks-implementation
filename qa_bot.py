from tensorflow import keras
import os, pickle, numpy
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_question_and_run_model():
    with open('train_qa.txt','rb') as file:
        train_data = pickle.load(file)
    print('----------------------------------------------------------------------')
    print(f'Train data len: {len(train_data)}')

    with open('test_qa.txt','rb') as file:
        test_data = pickle.load(file)
    print('----------------------------------------------------------------------')
    print(f'Test data len: {len(test_data)}')
    print('----------------------------------------------------------------------')

    all_data = test_data + train_data


    stories = []
    questions = []

    for story, question, answer in all_data:
        stories.append(len(story))
        questions.append(len(question))

    max_story_len = max(stories)
    max_question_len = max(questions)

    print('\nMax story length is:', max_story_len,'words')
    print('Max question length is:',max_question_len,'words')

    vocabulary = set()

    # add unique words to vocabulary
    for story, question, answer in all_data:
        vocabulary = vocabulary.union(set(story))
        vocabulary = vocabulary.union(set(question))

    vocabulary.add('no')
    vocabulary.add('yes')

    vocab_size = len(vocabulary)+1 # +1 because in keras paddind function it is required to have a placeholder

    print('\nTotal number of unique words in questions and stories:',vocab_size-1)
    print('----------------------------------------------------------------------')


    def vectorize_data(data,
                        word_index,
                        max_story_len,
                        max_question_len):

        stories = []
        questions = []

        answers = []

        for story, question, answer in data:
            stories_part = [word_index[word.lower()] for word in story]         # return index of each word according to their position in word index for stories
            questions_part = [word_index[word.lower()] for word in question]    # return index of each word according to their position in word index for questions

            answers_part = np.zeros(len(word_index)+1)                          # placeholder
            answers_part[word_index[answer]] = 1                                # in the index position of 'yes' or 'no' put 1 

            stories.append(stories_part)
            questions.append(questions_part)
            answers.append(answers_part)

        return (pad_sequences(stories,maxlen=max_story_len),pad_sequences(questions,maxlen=max_question_len),np.array(answers))     # return padded data


    qa_tokenizer = pd.read_csv('model/tokenizer.csv',dtype={'word':str})
    qa_tokenizer = dict(list(zip(qa_tokenizer['word'],qa_tokenizer['word_index'])))

    model = keras.models.load_model('model/memory_network.h5')


    print('----------------------------------------------------------------------')
    def generate_question():
        import random

        x = random.randint(a=1,b=len(test_data))

        print('\nStory:',' '.join(test_data[x-1][0]))
        s = test_data[x-1][0]
        print('\nQuestion:',' '.join(test_data[x-1][1]))
        q = test_data[x-1][1]
        print('\nAnswer:',test_data[x-1][2])
        a = test_data[x-1][2]

        return s, q, a, [(test_data[x-1])]


    s,q,a,question = generate_question()

    my_story,my_ques,my_ans = vectorize_data(question, qa_tokenizer,max_story_len,max_question_len)

    pred_results = model.predict(([ my_story, my_ques]))

    yes_prob = pred_results[0][qa_tokenizer['yes']]
    no_prob = pred_results[0][qa_tokenizer['no']]

    if yes_prob > no_prob:
        k = 'yes'
        prob = yes_prob
    else:
        k = 'no'
        prob = no_prob

    print("\nPredicted answer is: ", k)
    print("Probability of certainty was: ", round(prob*100,2),"%")
    print('----------------------------------------------------------------------')

    return s,q,a,k,round(prob*100,2)




def get_question():
    with open('train_qa.txt','rb') as file:
        train_data = pickle.load(file)
    print('----------------------------------------------------------------------')
    print(f'Train data len: {len(train_data)}')

    with open('test_qa.txt','rb') as file:
        test_data = pickle.load(file)
    print('----------------------------------------------------------------------')
    print(f'Test data len: {len(test_data)}')
    print('----------------------------------------------------------------------')

    all_data = test_data + train_data


    stories = []
    questions = []

    for story, question, answer in all_data:
        stories.append(len(story))
        questions.append(len(question))

    max_story_len = max(stories)
    max_question_len = max(questions)

    print('\nMax story length is:', max_story_len,'words')
    print('Max question length is:',max_question_len,'words')

    vocabulary = set()

    # add unique words to vocabulary
    for story, question, answer in all_data:
        vocabulary = vocabulary.union(set(story))
        vocabulary = vocabulary.union(set(question))

    vocabulary.add('no')
    vocabulary.add('yes')

    vocab_size = len(vocabulary)+1 # +1 because in keras paddind function it is required to have a placeholder

    print('\nTotal number of unique words in questions and stories:',vocab_size-1)
    print('----------------------------------------------------------------------')

    def generate_question():
        import random

        x = random.randint(a=1,b=len(test_data))

        print('\nStory:',' '.join(test_data[x-1][0]))
        s = test_data[x-1][0]
        print('\nQuestion:',' '.join(test_data[x-1][1]))
        q = test_data[x-1][1]
        print('\nAnswer:',test_data[x-1][2])
        a = test_data[x-1][2]

        return s, q, a, [(test_data[x-1])]


    s,q,a,question_input = generate_question()


    print('----------------------------------------------------------------------')

    return s,q,a,question_input,max_story_len,max_question_len

def model_predict(question,max_story_len,max_question_len):

    def vectorize_data(data,
                        word_index,
                        max_story_len,
                        max_question_len):

        stories = []
        questions = []

        answers = []

        for story, question, answer in data:
            stories_part = [word_index[word.lower()] for word in story]         # return index of each word according to their position in word index for stories
            questions_part = [word_index[word.lower()] for word in question]    # return index of each word according to their position in word index for questions

            answers_part = np.zeros(len(word_index)+1)                          # placeholder
            answers_part[word_index[answer]] = 1                                # in the index position of 'yes' or 'no' put 1 

            stories.append(stories_part)
            questions.append(questions_part)
            answers.append(answers_part)

        return (pad_sequences(stories,maxlen=max_story_len),pad_sequences(questions,maxlen=max_question_len),np.array(answers))     # return padded data
        
    qa_tokenizer = pd.read_csv('model/tokenizer.csv',dtype={'word':str})
    qa_tokenizer = dict(list(zip(qa_tokenizer['word'],qa_tokenizer['word_index'])))

    model = keras.models.load_model('model/memory_network.h5')

    my_story,my_ques,my_ans = vectorize_data(question, qa_tokenizer,max_story_len,max_question_len)

    pred_results = model.predict(([ my_story, my_ques]))

    yes_prob = pred_results[0][qa_tokenizer['yes']]
    no_prob = pred_results[0][qa_tokenizer['no']]

    if yes_prob > no_prob:
        k = 'yes'
        prob = yes_prob
    else:
        k = 'no'
        prob = no_prob

    print("\nPredicted answer is: ", k)
    print("Probability of certainty was: ", round(prob*100,2),"%")

    return k,f'{round(prob*100,2)}% '