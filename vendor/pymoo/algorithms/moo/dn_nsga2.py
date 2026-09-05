"""Decision-space niching NSGA-II baseline.

Independent pymoo implementation of the environmental-selection idea in
Liang, Yue, and Qu (CEC 2016): rank by objective-space nondominance and use
decision-space crowding to truncate the splitting front.
"""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


class DecisionSpaceRankAndCrowding(RankAndCrowding):
    """NSGA-II survival with crowding calculated on normalized decisions."""

    def __init__(self, nds=None):
        super().__init__(nds=nds)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self._decision_crowding = get_crowding_function("cd")

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = np.asarray(pop.get("F"), dtype=float)
        X = np.asarray(pop.get("X"), dtype=float)
        scale = np.where(np.asarray(problem.xu) > np.asarray(problem.xl),
                         np.asarray(problem.xu) - np.asarray(problem.xl), 1.0)
        Xn = (X - np.asarray(problem.xl)) / scale
        survivors = []
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for rank, front in enumerate(fronts):
            crowding = self._decision_crowding.do(Xn[front], n_remove=0)
            for local, index in enumerate(front):
                pop[index].set("rank", rank)
                pop[index].set("crowding", crowding[local])

            remaining = n_survive - len(survivors)
            if len(front) > remaining:
                order = randomized_argsort(
                    crowding, order="descending", method="numpy",
                    random_state=random_state,
                )
                survivors.extend(front[order[:remaining]])
                break
            survivors.extend(front)

        return pop[survivors]


class DNNSGA2(NSGA2):
    """Decision-niched NSGA-II using equal-budget pymoo operators."""

    def __init__(self, *args, survival=None, **kwargs):
        if survival is None:
            survival = DecisionSpaceRankAndCrowding()
        super().__init__(*args, survival=survival, **kwargs)
