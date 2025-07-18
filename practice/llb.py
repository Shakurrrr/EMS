class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return f"Deposited {amount} to {self.owner}'s account. New balance: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return f"Insufficient balance. Current balance: {self.balance}"
        self.balance -= amount

    def get_details(self):
        return f"Owner: {self.owner}, Balance: {self.balance}"

acc = BankAccount("Yusuf")
acc.deposit(1000)
acc.withdraw(500)
print(acc.get_details()) 


# Asking a new user what kinda cars they want to buy..

car_owner = input(f"what kind of car do you want sir??")
customer = ("mercedes", "bmw", "ferrari")
if car_owner.lower() in customer:
    print(f"Congratulations! {car_owner.title()} is available for purchase.")
else:
    print(f"Sorry, we do not have {car_owner.title()} available at the moment.")


