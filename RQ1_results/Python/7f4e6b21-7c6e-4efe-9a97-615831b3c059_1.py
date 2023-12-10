from flask import Blueprint
from .envapi import validation, database

locations_blueprint = Blueprint('locations', __name__)

@locations_blueprint.route('/locations', methods=['POST'])  
def create_location():
    ...
# Then in your environment.py, you register this blueprint

app.register_blueprint(locations_blueprint)
