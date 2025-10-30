import os
import json
from typing import List, Iterable, Set, Dict, Optional

import numpy as np
from implicit.als import AlternatingLeastSquares

class ALSRecommender:
    """
    Alternating Least Squares (ALS) Recommender.

    This class contains the core recommendation logic and is decoupled
    from file I/O. It can be instantiated directly with model components
    for easy testing or loaded from artifacts using the `load` classmethod.
    """
    def __init__(
        self, 
        user_factors: np.ndarray, 
        item_factors: np.ndarray, 
        user_map: Dict[str, int], 
        movie_inv_map: Dict[int, str]
    ):
        """
        Initializes the recommender with pre-loaded model components.
        """
        print("Initializing ALSRecommender...")
        self.model = AlternatingLeastSquares()
        self.model.user_factors = user_factors
        self.model.item_factors = item_factors
        
        self.user_map = user_map
        self.movie_inv_map = movie_inv_map
        print("ALSRecommender ready.")

    @classmethod
    def load(cls, artifacts_dir: str = "artifacts"):
        """
        Loads the model and mappings from disk and returns an instance.
        """
        print(f"Loading ALS model artifacts from {artifacts_dir}...")
        model_file = os.path.join(artifacts_dir, "als_model.npz")
        user_map_file = os.path.join(artifacts_dir, "user_map.json")
        movie_map_file = os.path.join(artifacts_dir, "movie_map.json")

        npz_file = np.load(model_file)
        user_factors = npz_file['user_factors']
        item_factors = npz_file['item_factors']

        with open(user_map_file, 'r') as f:
            user_map: Dict[str, int] = json.load(f)
        
        with open(movie_map_file, 'r') as f:
            movie_map_raw: Dict[str, int] = json.load(f)
            movie_inv_map: Dict[int, str] = {v: k for k, v in movie_map_raw.items()}

        return cls(user_factors, item_factors, user_map, movie_inv_map)

    def recommend(
        self,
        user_id: int,
        k: int = 20,
        exclude: Optional[Iterable[str]] = None
    ) -> List[str]:
        """
        Recommend top-k movies for a given user.
        """
        # Guard against asking for 0 recommendations, which can cause errors in the underlying library
        if k == 0:
            return []

        user_id_str = str(user_id)
        if user_id_str not in self.user_map:
            return []  # Cold start

        user_idx = self.user_map[user_id_str]
        
        recs = self.model.recommend(
            userid=user_idx,
            user_items=np.empty((0,0)),
            N=k + len(exclude or []),
            filter_already_liked_items=False
        )

        banned: Set[str] = set(map(str, exclude)) if exclude else set()
        
        top_movies = []
        for item_idx, score in zip(*recs):
            movie_id = self.movie_inv_map.get(item_idx)
            if movie_id and movie_id not in banned:
                top_movies.append(movie_id)

        return top_movies[:k]
