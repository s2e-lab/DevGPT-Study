class BankAccount:
    def __init__(self):
        self.__balance = 0.0
    
    def __deduct_fees(self):
        # Some code to deduct fees from the balance
        pass
    
    def deposit(self, amount):
        self.__balance += amount
    
    def get_balance(self):
        return self.__balance

account = BankAccount()
account.deposit(1000)  # Accessing 'private' method through name mangling
print(account.get_balance())  # Accessing 'private' variable through name mangling
# print(account.__balance)  # AttributeError: 'BankAccount' object has no attribute '__balance'
# account.__deduct_fees()  # AttributeError: 'BankAccount' object has no attribute '__deduct_fees'
