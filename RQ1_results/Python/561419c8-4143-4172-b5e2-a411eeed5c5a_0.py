import random

class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']
        self.cards = [{'suit': suit, 'value': value} for suit in suits for value in values]
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def add_card(self, card):
        self.hand.append(card)

    def get_hand_value(self):
        value = 0
        aces = 0
        for card in self.hand:
            if card['value'] in ['Jack', 'Queen', 'King']:
                value += 10
            elif card['value'] == 'Ace':
                value += 11
                aces += 1
            else:
                value += int(card['value'][0])

        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def show_hand(self, hide_first_card=False):
        if hide_first_card:
            print(f"{self.name}'s hand: [???, {self.hand[1]['value']} of {self.hand[1]['suit']}] and {len(self.hand) - 2} more cards" if len(self.hand) > 2 else "")
        else:
            hand_str = ', '.join([f"{card['value']} of {card['suit']}" for card in self.hand])
            print(f"{self.name}'s hand: {hand_str}")

def blackjack():
    deck = Deck()
    player = Player("Player")
    dealer = Player("Dealer")

    # Initial dealing
    for _ in range(2):
        player.add_card(deck.deal())
        dealer.add_card(deck.deal())

    player.show_hand()
    dealer.show_hand(hide_first_card=True)

    # Player's turn
    while player.get_hand_value() < 21:
        action = input("Do you want to (H)it or (S)tand? ").lower()
        if action == 'h':
            player.add_card(deck.deal())
            player.show_hand()
        elif action == 's':
            break

    # Dealer's turn
    while dealer.get_hand_value() < 17:
        dealer.add_card(deck.deal())

    # Show results
    player.show_hand()
    dealer.show_hand()

    if player.get_hand_value() > 21:
        print("Player busts! Dealer wins!")
    elif dealer.get_hand_value() > 21:
        print("Dealer busts! Player wins!")
    elif player.get_hand_value() > dealer.get_hand_value():
        print("Player wins!")
    elif dealer.get_hand_value() > player.get_hand_value():
        print("Dealer wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    blackjack()
