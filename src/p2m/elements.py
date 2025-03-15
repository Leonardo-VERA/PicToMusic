from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Optional

@dataclass
class StaffLine:
    """Represents a staff line and its associated notes in a music score."""
    index: int 
    line_contour: np.ndarray 
    bounds: Tuple[int, int, int, int]
    notes: List['Note'] = field(default_factory=list)
    key: Optional['Key'] = None
    
@dataclass
class Key:
    """Represents a key signature in a music score."""
    line_index: int
    contour: np.ndarray
    bounds: Tuple[int, int, int, int]
    relative_position: Tuple[int, int] 
    absolute_position: Tuple[int, int] 
    metric: Tuple[int, int] = (4, 4)
    label: Optional[str] = None
    gamme: Optional[str] = None

@dataclass
class Note:
    """Represents a musical element in the score."""
    index: int
    relative_index: int
    line_index: int
    contour: np.ndarray 
    bounds: Tuple[int, int, int, int] 
    relative_position: Tuple[int, int] 
    absolute_position: Tuple[int, int] 
    label: Optional[str] = None