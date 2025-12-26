# Flask Application Setup

This repository includes a Flask web application that provides a landing page for all the machine learning projects.

## Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### On Windows:
```bash
set FLASK_APP=main.py
flask run
```

#### On Linux/Mac:
```bash
export FLASK_APP=main.py
flask run
```

The application will start on `http://127.0.0.1:5000`

### Alternative: Run directly with Python
```bash
python main.py
```

## Features

- **Landing Page**: Showcases all 5 machine learning projects with descriptions and technologies
- **About Page**: Explains the repository structure and ML pipeline
- **Responsive Design**: Mobile-friendly interface
- **Easy Configuration**: Project data managed in `projects.json`

## Project Structure

```
├── main.py              # Flask application
├── requirements.txt     # Python dependencies
├── projects.json        # Project configuration
├── templates/           # HTML templates
│   ├── index.html      # Landing page
│   └── about.html      # About page
└── static/             # Static assets
    └── style.css       # Stylesheet
```

## Adding New Projects

To add a new project to the landing page, edit `projects.json`:

```json
{
    "name": "Project Name",
    "description": "Project description",
    "folder": "Project Folder Name",
    "technologies": ["Tech1", "Tech2"]
}
```

## Production Deployment

For production use, deploy with a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn main:app
```

Or with uWSGI:

```bash
pip install uwsgi
uwsgi --http :5000 --wsgi-file main.py --callable app
```

## Security Notes

- Debug mode is disabled by default
- Uses Flask 3.0.3 with latest security patches
- UTF-8 encoding for cross-platform compatibility

## Contributing

Feel free to add new projects or improve the web application!
