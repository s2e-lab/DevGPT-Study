class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def to_dict(self):
        return {'name': self.name, 'age': self.age}

person = Person("Alice", 30)
person_dict = person.to_dict()
print(person_dict)  # Outputs: {'name': 'Alice', 'age': 30}
