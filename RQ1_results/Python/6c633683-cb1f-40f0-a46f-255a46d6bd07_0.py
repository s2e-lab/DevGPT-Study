from hello_flask_app import app, Person
with app.app_context():
    print(Person.query.all())
