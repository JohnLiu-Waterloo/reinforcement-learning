# section 5.3, Monte Carlo control

import numpy as np
import random

class Deck:
    def __init__(self):
        self.ncards = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 16]

    def draw(self):
        prob = [c / sum(self.ncards) for c in self.ncards] 
        card = random.choices(range(11), prob)[0]
        # self.ncards[card] -= 1
        return card

def update_state(states, r):
    for s in states:
        a = 1 if s[0] else 0
        N[a][s[1]][s[2]][s[3]] += 1
        R[a][s[1]][s[2]][s[3]] += r

episodes = 1000000
# episodes = 5
DEBUG = False

N = np.zeros((2, 22, 11, 2))
R = np.zeros((2, 22, 11, 2))

for e in range(episodes):
    if e % 10000 == 0 and e > 0:
        print("episode ", e)
    deck = Deck()
    c1 = deck.draw() 
    c2 = deck.draw()

    val = c1 + c2
    usable_ace = (c1 == 1 or c2 == 1) and val <= 11
    val = val + 10 if usable_ace else val

    d1 = deck.draw()
    d2 = deck.draw()    # hidden

    player_cards = [c1, c2]
    dealer_cards = [d1, d2]

    # player's turn
    states = []
    while val <= 21:
        # policy description
        # always hit if val <= 10
        # if val >= 11, then hit more frequently the smaller val is
        should_hit = random.random() < (21-val)/11
        if not should_hit:
            states.append((usable_ace, val, d1, 0))
            break
        else:
            states.append((usable_ace, val, d1, 1))
            c = deck.draw()
            val += c
            if val > 21 and usable_ace:
                val -= 10
                usable_ace = False
            elif c == 1 and val <= 11:
                val += 10
                usable_ace = True
            player_cards.append(c)

    if val > 21:
        if DEBUG:
            print(player_cards)
            print(val)
            print(states)
            print(dealer_cards)
            print()
        update_state(states, -1)
        continue

    # dealer's turn
    dealer_val = d1 + d2 
    dealer_ace = d1 == 1 or d2 == 1
    while dealer_val < 17:
        if dealer_ace and 7 <= dealer_val <= 11:
            dealer_val = dealer_val + 10
            break

        c = deck.draw()
        dealer_val += c
        dealer_ace = dealer_ace or c == 1
        dealer_cards.append(c)

    if DEBUG:
        print(player_cards)
        print(states)
        print(val)
        print(dealer_cards)
        print(dealer_val)
        print()

    if dealer_val > 21 or dealer_val < val:
        update_state(states, 1)
    elif val == dealer_val:
        update_state(states, 0)
    else:
        update_state(states, -1)

for ace in range(2):
    print("No usable ace" if ace == 0 else "Usable ace")
    for i in range(21, 10, -1):
        actions = []
        for j in range(1, 11):
            if (np.sum(N[ace][i][j])) > 0:
                stay_reward = R[ace][i][j][0] / N[ace][i][j][0]
                hit_reward = R[ace][i][j][1] / N[ace][i][j][1]
                actions.append(hit_reward > stay_reward)
                if DEBUG:
                    print(i, " ", j, " ", " ", round(stay_reward, 2), " ", round(hit_reward, 2))
        print(i, ": ", actions)
    print()

for i, j in [(12, 2), (12, 3)]:
    stay_reward = R[0][i][j][0] / N[0][i][j][0]
    hit_reward = R[0][i][j][1] / N[0][i][j][1]
    print("stay: ", stay_reward, ", hit: ", hit_reward)