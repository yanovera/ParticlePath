from pydantic import BaseModel
from numpy.linalg import norm

BeaconID = int


class WorldLimits(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max


class Point(BaseModel):
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return norm((self.x - other.x, self.y - other.y))


class Particle(Point):
    pass


class Beacon(Point):
    id: BeaconID


class Agent(Point):
    pass


class OdometerReading(BaseModel):
    vx: float
    vy: float


class Variances(BaseModel):
    odometer: float = 0.1
    proximity: float = 0.4
    motion: float = 0.001
    regulation = 0.01
