class Car:
    def __init__(self):
        self.brand = ""
    
    def start(self):
        print("Car started.")

my_car = Car()
my_car.brand = "Toyota"  # Accessing public variable
my_car.start()  # Accessing public method
