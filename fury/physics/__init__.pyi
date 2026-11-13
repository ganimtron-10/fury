# This file is used to enable type hinting for the `fury.physics` module.
# explicit definition of `__all__` will enable type inference for engines.

__all__ = [
    "Body",
    "BodySet",
    "Sphere",
    "World",
]

from .bodies import Body, BodySet
from .shapes import Sphere
from .world import World
