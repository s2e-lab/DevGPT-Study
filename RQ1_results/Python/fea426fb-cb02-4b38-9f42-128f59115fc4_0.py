from random import randint

class Dice(object):
    def __init__(self):
        self.value = ()
    
    def roll_dice(self):
        self.value = (randint(1, 6), randint(1, 6))
        return self.value
