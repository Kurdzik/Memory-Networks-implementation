from crypt import methods
from flask import Flask,render_template, request
from qa_bot import get_question,model_predict
import pandas as pd



app = Flask(__name__,template_folder='./templates',static_folder='./templates/static')

story = ''
question = ''
correct_answer = ''
pred_answer = ''
pred_conf = ''
warning = ''
question_input = ''
max_story_len = 156
max_question_len = 6


@app.route('/')
def main():

    story = ''
    question = ''
    correct_answer = ''
    pred_answer = ''
    pred_conf = ''

    return render_template('index.html',story=story,question=question,correct_answer=correct_answer,pred_answer=pred_answer,pred_conf=pred_conf)



@app.route('/get_question')
def render_question():
    global story,question,correct_answer,question_input,max_story_len,max_question_len

    story,question,correct_answer,question_input,max_story_len,max_question_len = get_question()
    
    story = ' '.join(story)
    question = ' '.join(question)
    pred_answer = ''
    pred_conf = ''

    return render_template('index.html',story=story,question=question,correct_answer=correct_answer,pred_answer=pred_answer,pred_conf=pred_conf)



@app.route('/predict')
def predict():
    global story,question,correct_answer,question_input,max_story_len,max_question_len
    
    pred_answer,pred_conf = model_predict(question=question_input,max_question_len=max_question_len,max_story_len=max_story_len)

    return render_template('index.html',story=story,question=question,correct_answer=correct_answer,pred_answer=pred_answer,pred_conf=pred_conf)






@app.route('/ask_question')
def get_cutom_question():
    global story,question,correct_answer,question_input,max_story_len,max_question_len

    vocab = list(pd.read_csv('model/vocabulary.csv')['word'])
    story = ''
    question = ''
    correct_answer = ''
    pred_answer = ''
    pred_conf = ''
    warning = ''
    missmached_words = ''

    return render_template('index2.html',story=story,question=question,correct_answer=correct_answer,pred_answer=pred_answer,pred_conf=pred_conf,vocab=vocab, warning=warning,missmached_words = missmached_words)



@app.route('/predict_answer',methods=['POST','GET'])
def predict_custom_question():
    global story,question,correct_answer,question_input,max_story_len,max_question_len,warning

    vocab = list(pd.read_csv('model/vocabulary.csv')['word'])
    story = request.form['input_story']
    saved_story = request.form['input_story']
    question = request.form['input_question']
    saved_question = request.form['input_question']
    correct_answer = ''
    pred_answer = ''
    pred_conf = ''
    



    # Check criteria for inputs

    missmached_words = []

    for word in story.split(' '):
        if word not in vocab and isinstance(story.split(' '),list):
            missmached_words.append(word)
            warning_s = ''
        elif not isinstance(story.split(' '),list):
            warning_s = "Story can't be one word"
        else: 
            continue
    
    for word in question.split(' '):
        if word not in vocab and isinstance(question.split(' '),list):
            missmached_words.append(word)
            warning_q = ''
        elif not isinstance(question.split(' '),list):
            warning_q = "Question can't be one word"
        else:
            continue


    if len(missmached_words)==0:
        missmached_words = ''
        warning_s = ''
        warning_q = ''





    # Make predictions if question and answer meet the criteria

    if len(story)!=0 or len(question)!=0 or len(missmached_words)!=0:
        try:
            question_input = [(story.split(' '),question.split(' '),'yes')]

            pred_answer,pred_conf = model_predict(question=question_input,max_question_len=max_question_len,max_story_len=max_story_len) 
            print(pred_answer)
            warning = ''
        except Exception:
            if len(missmached_words) > 0: 
                warning = 'Please enter story and question in a correct format, following words are not in a vocabulary:'
            else:
                warning = f'{warning_s} {warning_q}'

    else:
        warning = 'Please enter story and question first'


    return render_template('index2.html',story=story,
                                         question=question,
                                         correct_answer=correct_answer,
                                         pred_answer=pred_answer,
                                         pred_conf=pred_conf,
                                         vocab=vocab, 
                                         warning=warning,
                                         missmached_words=missmached_words,
                                         saved_question = saved_question,
                                         saved_story = saved_story)



if __name__ == '__main__':
    app.run()
