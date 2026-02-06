import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class FitResult:
    name: str
    imputer: str
    scaler: str
    is_tuned: bool
    best_params: Optional[Dict[str, Any]]
    cv_rmse: Optional[float]
    val_rmse: float
    val_mae: float
    val_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float
