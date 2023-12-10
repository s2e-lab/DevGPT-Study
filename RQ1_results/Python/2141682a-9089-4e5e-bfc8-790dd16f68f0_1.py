class Animal:
    def __init__(self):
        self._name = ""
    
    def _make_sound(self):
        print("Animal makes a sound.")

class Dog(Animal):
    def greet(self):
        print("Dog barks, says hello!")
        self._make_sound()  # Accessing 'protected' method through convention

my_dog = Dog()
my_dog._name = "Buddy"  # Accessing 'protected' variable through convention
my_dog.greet()  # Accessing 'protected' method through convention
