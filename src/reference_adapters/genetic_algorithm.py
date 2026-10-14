"""Depth-B binary feature search, adapted from ML-From-Scratch (MIT).

Reference snapshot: docs/reference/genetic_algorithm.py, Erik Linder-Noren.
The population -> fitness -> selection -> crossover -> mutation loop is retained.
Changes: binary masks and callback fitness, tournament selection, seeded RNG,
empty-mask repair, caching, optional elitism, diversity trace, all-time best.
"""
import numpy as np


class FeatureGA:
    def __init__(self, n_features, population_size=16, mutation_rate=None,
                 generations=12, seed=42, elite_count=1):
        if n_features < 2 or population_size < 2 or generations < 1:
            raise ValueError("At least two features/individuals and one generation are required")
        if not 0 <= elite_count < population_size:
            raise ValueError("elite_count must be smaller than population_size")
        self.n_features, self.population_size = n_features, population_size
        self.mutation_rate = 1 / n_features if mutation_rate is None else mutation_rate
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be a probability")
        self.generations, self.seed, self.elite_count = generations, seed, elite_count

    def _repair(self, mask):
        mask = mask.copy()
        if not mask.any():
            mask[self.rng_.integers(self.n_features)] = True
        return mask

    def _initialize(self):
        return np.array([self._repair(mask) for mask in
                         self.rng_.integers(0, 2, size=(self.population_size, self.n_features)).astype(bool)])

    def _mutate(self, mask):
        return self._repair(mask ^ (self.rng_.random(self.n_features) < self.mutation_rate))

    def _crossover(self, parent1, parent2):
        cut = self.rng_.integers(1, self.n_features)
        return (np.concatenate([parent1[:cut], parent2[cut:]]),
                np.concatenate([parent2[:cut], parent1[cut:]]))

    def _select(self, population, scores):
        candidates = self.rng_.choice(len(population), size=min(3, len(population)), replace=False)
        return population[candidates[np.argmax(scores[candidates])]]

    def run(self, fitness):
        self.rng_ = np.random.default_rng(self.seed)
        self.cache_, self.history_ = {}, []
        population = self._initialize()
        self.initial_population_ = population.copy()
        self.best_fitness_, self.best_mask_ = -np.inf, None
        for generation in range(self.generations):
            scores = []
            for mask in population:
                key = tuple(map(bool, mask))
                if key not in self.cache_:
                    self.cache_[key] = float(fitness(mask))
                scores.append(self.cache_[key])
            scores = np.asarray(scores)
            if not np.isfinite(scores).all():
                raise ValueError("Fitness must be finite")
            top = int(np.argmax(scores))
            if scores[top] > self.best_fitness_:
                self.best_fitness_ = float(scores[top])
                self.best_mask_ = population[top].copy()
            # Mean pairwise Hamming distance, normalized by chromosome length.
            diversity = np.mean([np.mean(population[i] != population[j])
                                 for i in range(len(population)) for j in range(i + 1, len(population))])
            self.history_.append({
                "generation": generation, "generation_best": float(scores[top]),
                "best_so_far": self.best_fitness_, "mean_fitness": float(scores.mean()),
                "diversity": float(diversity),
                "unique_fraction": len({tuple(mask) for mask in population}) / len(population),
            })
            if generation == self.generations - 1:
                break
            elites = np.argsort(-scores, kind="stable")[:self.elite_count]
            next_population = [population[i].copy() for i in elites]
            while len(next_population) < self.population_size:
                children = self._crossover(self._select(population, scores), self._select(population, scores))
                for child in children:
                    if len(next_population) < self.population_size:
                        next_population.append(self._mutate(child))
            population = np.asarray(next_population)
        return self
