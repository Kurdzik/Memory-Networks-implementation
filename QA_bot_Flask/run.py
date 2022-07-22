from flask import Flask,render_template, request
from qa_bot import get_question,model_predict



app = Flask(__name__,template_folder='./templates',static_folder='./templates/static')

story = ''
question = ''
correct_answer = ''
pred_answer = ''
pred_conf = ''
question_input = ''
max_story_len = ''
max_question_len = ''


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



if __name__ == '__main__':
    app.run(host='localhost',port=4040)