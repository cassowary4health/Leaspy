#!/usr/bin/python3

from flask import Flask
from flask import render_template, request
import json




application = Flask('leaspy', template_folder='templates')
application._static_folder = 'static/'

@application.route("/")
def index():
    return render_template('index.html')



@application.route("/", methods=['POST'])
def personalize():
    from models.personalize import get_individual_parameters

    data = request.get_json()
    individual_parameters = get_individual_parameters(data)
    return json.dumps(individual_parameters)

if __name__ == "__main__":
    application.run()
