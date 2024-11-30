import numpy as np



"""
features

buying
bidding


"""

class Game():

  def __init__(self):

    self.board_squares = [i for i in range(30)]

  def dice_roll(self):

    def roll():
      return np.random.choice([i for i in range(1,7)], 1, [1/6 for i in range(6)])

    dice1 = roll()
    dice2 = roll()

    return (dice1, dice2)



  def simulate(self):

    return a