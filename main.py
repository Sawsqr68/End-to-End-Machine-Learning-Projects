from flask import Flask, render_template
import json
import os

app = Flask(__name__)

# Load project data once at startup
PROJECTS_FILE = os.path.join(os.path.dirname(__file__), 'projects.json')
with open(PROJECTS_FILE, 'r', encoding='utf-8') as f:
    PROJECTS = json.load(f)

@app.route('/')
def index():
    """Landing page showing all ML projects"""
    return render_template('index.html', projects=PROJECTS)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

if __name__ == '__main__':
    # Note: debug mode should only be used in development
    # In production, use a WSGI server like gunicorn or uwsgi
    app.run()
