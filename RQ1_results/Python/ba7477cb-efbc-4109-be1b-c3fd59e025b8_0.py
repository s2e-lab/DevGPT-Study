class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades'] for rank in range(1, 14)]

    def shuffle(self):
        # Code to shuffle the deck

    def draw_card(self):
        # Code to draw a card

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def value(self):
        # Code to calculate value

    def is_blackjack(self):
        # Code to check for blackjack

    def is_bust(self):
        # Code to check for bust

class Player:
    def __init__(self):
        self.hand = Hand()

    def hit(self, deck):
        # Code to draw a card

    def stand(self):
        # Code to stand

class Dealer(Player):
    def play(self, deck):
        # Code to draw cards until hand value is at least 17

class Game:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()
        self.deck = Deck()

    def play_round(self):
        # Code to play a round

    def check_winner(self):
        # Code to check who won

    def reset(self):
        # Code to reset for another round
