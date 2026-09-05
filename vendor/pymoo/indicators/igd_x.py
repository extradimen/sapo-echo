"""Decision-space inverted generational distance for multimodal MOO."""

from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance


class IGDX(DistanceIndicator):
    """Mean distance from a reference Pareto set to the returned decisions.

    Decision variables are normalized with the problem bounds when ``xl`` and
    ``xu`` are supplied, making results comparable across differently scaled
    variables.
    """

    def __init__(self, ps, xl=None, xu=None, **kwargs):
        zero_to_one = xl is not None and xu is not None
        super().__init__(
            ps,
            euclidean_distance,
            axis=1,
            zero_to_one=zero_to_one,
            ideal=xl,
            nadir=xu,
            **kwargs,
        )
