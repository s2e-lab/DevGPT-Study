class Animal:
    def sound(self):
        raise NotImplementedError()

class Dog(Animal):
    def sound(self):
        return "Woof"

animal = Dog()
print(animal.sound()) # Output: Woof
