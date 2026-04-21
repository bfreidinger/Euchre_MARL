from rlcard.utils.euchre_utils import is_left, is_right, NON_TRUMP

class EuchreJudger(object):

    def __init__(self):
        pass

    def judge_trick(self, game):
        center_cards = game.center
        trump = game.trump
        leader = game.current_player
        player_order = self._get_player_order(leader)
        center_cards = {}
        for i,p in enumerate(player_order):
            center_cards[p] = game.center[i]

        lead_suit = center_cards[leader].suit
        winning_card = center_cards[leader]
        
        if is_right(winning_card, trump):
            return [k for k, v in center_cards.items() if winning_card == v][0]

        for player in player_order[1:]:
            candidate_card = center_cards[player]
            
            if is_right(candidate_card, trump):
                winning_card = candidate_card
                break

            if is_left(winning_card, trump):
                continue

            if is_left(candidate_card, trump):
                winning_card = candidate_card
                continue

            if (candidate_card.suit != trump):
                if candidate_card.suit == lead_suit:
                    if NON_TRUMP.index(candidate_card.rank) > NON_TRUMP.index(winning_card.rank):
                        winning_card = candidate_card
                        continue
            
            if (candidate_card.suit == trump):
                if candidate_card.suit != winning_card.suit:
                    winning_card = candidate_card
                    continue
                if NON_TRUMP.index(candidate_card.rank) > NON_TRUMP.index(winning_card.rank):
                    winning_card = candidate_card
                    continue
        
        return [k for k, v in center_cards.items() if winning_card == v][0]

    def judge_hand(self, game):
        caller = game.calling_player
        maker_team    = [caller, (caller + 2) % 4]
        defender_team = [(caller + 1) % 4, (caller + 3) % 4]
        maker_tricks  = game.score[maker_team[0]] + game.score[maker_team[1]]
        if maker_tricks == 5:      # march
            return maker_team, 2
        elif maker_tricks >= 3:    # made it
            return maker_team, 1
        else:                      # euchred
            return defender_team, 2

    def _get_player_order(self, leader):
        return [(i+leader)%4 for i in range(4)]
