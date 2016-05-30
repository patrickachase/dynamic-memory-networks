from flask import render_template
from app import app
from flask import request

from dmn_with_word_embedding import add_placeholders, convert_to_vectors, RNN, \
  input_module, question_module, episodic_memory_module, answer_module


def get_answer(input, question):
  return "yes"

@app.route('/')
@app.route('/dmn_demo')
def index():
    input = request.args.get('input')
    question = request.args.get('question')

    return render_template("dmn_demo.html",
                           input=input,
                           question=question)

