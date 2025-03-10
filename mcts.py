import math
import numpy as np

# Concerns: Add epsilon amount to UCB evaluation to ensure probability is considered
# Caveat: Q in heuristic might obviate this.
# Concerns: No Dir noise being added. If it is added, tests would break.
# Caveat: Make Dir a switch, write tests that use Dir with fixed seed.


# An efficient, vectorized Monte Carlo tree search implementation.
# Uses no loops, done completely with numpy.
class MCTS():

    def __init__(self, game, nn):
        self.game = game
        self.nn = nn
        self.tree = {}

    # Produces a hash-friendly representation of an ndarray.
    # This is used to index nodes in the accumulated Monte Carlo tree.
    def np_hash(self, data):
        return data.tostring()

    # Run a MCTS simulation starting from state s of the tree.
    # The tree is accumulated in the self.tree dictionary.
    # The epsilon fix prevents the U term from being 0 when unexplored (N=0).
    # With the fix, priors (P) can be factored in immediately during selection and expansion.
    # This makes the search more efficient, given there are strong priors.
    def simulate(self, s, cpuct=1, epsilon_fix=True):
        hashed_s = self.np_hash(s) # Key for state in dictionary
        current_player = self.game.get_player(s)
        if hashed_s in self.tree: # Not at leaf; select.
            stats = self.tree[hashed_s]
            N, Q, P = stats[:,1], stats[:,2], stats[:,3]
            U = cpuct*P*math.sqrt(N.sum() + (1e-6 if epsilon_fix else 0))/(1 + N)
            heuristic = Q + U
            best_a_idx = np.argmax(heuristic)
            best_a = stats[best_a_idx, 0] # Pick best action to take
            template = np.zeros_like(self.game.get_available_actions(s)) # Submit action to get s'
            template[tuple(best_a)] = True
            s_prime = self.game.take_action(s, template)
            v, winning_player = self.simulate(s_prime) # Forward simulate with this action
            n, q = N[best_a_idx], Q[best_a_idx]
            adj_v = v if current_player == winning_player else -v
            stats[best_a_idx, 2] = (n*q+adj_v)/(n + 1)
            stats[best_a_idx, 1] += 1
            return v, winning_player

        else: # Expand
            w = self.game.check_winner(s)
            if w is not None: # Reached a terminal node
                return 1 if w != -1 else 0, w # Someone won, or tie
            available_actions = self.game.get_available_actions(s)
            idx = np.stack(np.where(available_actions)).T
            p, v = self.nn.predict(s)

            # NOTE: the shape is NOT (7, .) because we only save info for legal actions
            stats = np.zeros((len(idx), 4), dtype=object)
            stats[:,-1] = p
            stats[:,0] = list(idx)
            self.tree[hashed_s] = stats
            return v, current_player


    # Returns the MCTS policy distribution for state s.
    # The temperature parameter softens or hardens this distribution.
    def get_distribution(self, s, temperature):
        hashed_s = self.np_hash(s)
        stats = self.tree[hashed_s][:,:2].copy()
        N = stats[:,1]
        try:
            raised = np.power(N, 1/temperature)
        # As temperature approaches 0, the effect becomes equivalent to argmax.
        except (ZeroDivisionError, OverflowError):
            raised = np.zeros_like(N)
            raised[N.argmax()] = 1

        total = raised.sum()
        # If all children are unexplored, prior is uniform.
        if total == 0:
            raised[:] = 1
            total = raised.sum()
        dist = raised/total
        stats[:,1] = dist
        return stats


