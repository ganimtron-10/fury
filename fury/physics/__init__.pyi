# This file is used to enable type hinting for the `fury.physics` module.
# explicit definition of `__all__` will enable type inference for engines.

__all__ = [
    "Body",
    "BodySet",
    "Box",
    "Sphere",
    "World",
]

from .bodies import Body, BodySet
from .shapes import Box, Sphere
from .world import World
