from dataclasses import dataclass

@dataclass
class derivs_interpolation:
    keypoint_method: str
    minN: int
    maxN: int
    jerk_threshold: float
    iterative_error_threshold: float

@dataclass
class index_tuple:
    start_index: int
    end_index: int