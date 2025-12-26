from flask import Flask, render_template
import json
import os

app = Flask(__name__)

def load_projects():
    """Load project data from JSON file"""
    projects_file = os.path.join(os.path.dirname(__file__), 'projects.json')
    with open(projects_file, 'r') as f:
        return json.load(f)

@app.route('/')
def index():
    """Landing page showing all ML projects"""
    projects = load_projects()
    return render_template('index.html', projects=projects)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
