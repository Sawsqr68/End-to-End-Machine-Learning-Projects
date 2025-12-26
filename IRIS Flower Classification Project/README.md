## IRIS FLOWER CLASSIFICATION

<p align="center"><img src="https://github.com/pratik-276/End-to-End-Machine-Learning-Projects/blob/master/IRIS%20Flower%20Classification%20Project/static/Iris-virginica.PNG" height="200" width="200"></p>

## Introduction

This Web Application powered by Machine Learning and Flask focusses on bringing every aspect of Iris Flower classification together. <b>Iris Flower Classification project</b> is one of the most basic projects in ML used to get a better understanding of data and how ML models work. This project tries to bring the data description, EDA, ML models performance and finally prediction model all under one roof.</b>

## Technology Used

<ul>
  <li>HTML</li>
  <li>CSS</li>
  <li>JS</li>
  <li>Flask</li>
  <li>Pandas</li>
  <li>Sklearn</li>
</ul>

## Visualizations
All the visualizations used in this project can be found in this <a href="https://www.kaggle.com/pratik1120/iris-visualization-and-model-performances">Kaggle Kernel</a>

## Installation

1. Drop a ⭐ on the Github Repository.
2. Download the repo as a zip file or clone it using the following command
```sh
https://github.com/pratik-276/End-to-End-Machine-Learning-Projects.git
```

3. Move inside the ` /IRIS Flower Classification Project ` folder and open CLI in that folder

4. Install the required libraries by the following command
```sh
pip install -r requirements.txt
```

5a. **Option 1: Run the Flask web application**
```sh
set FLASK_APP=main.py
flask run
```
Then go to `http://127.0.0.1:5000/` and check out the flask app.

5b. **Option 2: Use the CLI (Command Line Interface)**
```sh
# Interactive mode
python cli.py

# Or predict directly with measurements
python cli.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2

# Short form
python cli.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2
```


## Usage

### Web Interface
Run the Flask application to use the web interface:
```sh
set FLASK_APP=main.py
flask run
```
Then navigate to `http://127.0.0.1:5000/`

### Command Line Interface (CLI)
The project now includes a CLI for quick predictions:
```sh
# Interactive mode - you'll be prompted for measurements
python cli.py

# Direct prediction with arguments
python cli.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2

# Using short form arguments
python cli.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2

# Get help
python cli.py --help
```

## Contents

1. Data Description
2. EDA
<img src="https://github.com/pratik-276/End-to-End-Machine-Learning-Projects/blob/master/IRIS%20Flower%20Classification%20Project/static/readmeeda.PNG">
3. Model Performances
<img src="https://github.com/pratik-276/End-to-End-Machine-Learning-Projects/blob/master/IRIS%20Flower%20Classification%20Project/static/readmeperformance.PNG">
4. Prediction Model
<img src="https://github.com/pratik-276/End-to-End-Machine-Learning-Projects/blob/master/IRIS%20Flower%20Classification%20Project/static/readmemodel.PNG">

## Conclusion

This is a try to present a much explored dataset in a new way. Do drop a ⭐ if you like it. More such projects are coming soon which will involve more steps related to data cleaning, analysis, data wrngling and more.
