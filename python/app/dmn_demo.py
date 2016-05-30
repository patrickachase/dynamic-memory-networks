from flask import render_template
from app import app


@app.route('/')
@app.route('/dmn_demo')
def index():
    user = None
    posts = None
    return render_template("dmn_demo.html",
                           title='Home',
                           user=user,
                           posts=posts)

