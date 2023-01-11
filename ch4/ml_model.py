

from sklearn.model_selection import RandomizedSearchCV

class MlModel:

    def __init__(self,
                 ml_model,
                 parameters,
                 n_jobs,
                 scoring,
                 n_iter,
                 random_state):

        self.parameters = parameters
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.ml_model = ml_model

    def tune(self, X_features, y):

        self.clf = RandomizedSearchCV(self.ml_model,
                                      self.parameters,
                                      n_jobs=self.n_jobs,
                                      scoring = self.scoring,
                                      n_iter = self.n_iter,
                                      random_state = self.random_state)

        self.clf.fit(X_features, y)

    def predict(self, X_features):

        return self.clf.predict(X_features)