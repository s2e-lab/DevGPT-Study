class Vehicle:
    # Class attribute to keep track of the total number of vehicles
    total_vehicles = 0

    def __init__(self, brand):
        self.brand = brand
        Vehicle.total_vehicles += 1

    def __str__(self):
        return f"{self.brand} Vehicle"


class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

    def __str__(self):
        return f"{self.brand} {self.model} Car"


class Bike(Vehicle):
    def __init__(self, brand, type):
        super().__init__(brand)
        self.type = type

    def __str__(self):
        return f"{self.brand} {self.type} Bike"


car1 = Car("Toyota", "Corolla")
car2 = Car("Honda", "Civic")
bike1 = Bike("Harley-Davidson", "Cruiser")

print(f"Total number of vehicles: {Vehicle.total_vehicles}")  # Output: 3
print(f"Total number of vehicles: {Car.total_vehicles}")     # Output: 3
print(f"Total number of vehicles: {Bike.total_vehicles}")    # Output: 3
