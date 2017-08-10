import time
import json
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from sklearn.pipeline import Pipeline


class Classify:
    def __init__(self, courses_json_filename="courses.json"):
        self.courses = json.load(fp=open(courses_json_filename), object_pairs_hook=OrderedDict)
        self.departments = None
        self.training_set = None
        self.model = None

    def build_training_set(self):
        courselists = [(d["description"], d["department"],) for d in self.courses]
        self.training_set = courselists
        return self.training_set

    def train_model(self, find_best_params=False):
        texts = [tup[0] for tup in self.training_set]
        classes = [tup[1] for tup in self.training_set]

        parameters = {
            'vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),
            'vec__max_features': (None, 5000, 10000, 20000),
            'vec__min_df': (1, 5, 10, 20, 50),
            'tfidf__use_idf': (True, False),
            'tfidf__sublinear_tf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        }

        pipeline = Pipeline([
            ('vec', CountVectorizer(max_df=1.0, max_features=10000, min_df=1, stop_words='english')),
            ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=False, use_idf=True)),
            ('clf', MultinomialNB(alpha=0.01))
        ])

        if find_best_params:
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)
            t0 = time.time()
            grid_search.fit(texts, classes)
            print("done in {0}s".format(time.time() - t0))
            print("Best score: {0}".format(grid_search.best_score_))
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(list(parameters.keys())):
                print("\t{0}: {1}".format(param_name, best_parameters[param_name]))
        pipeline.fit(texts, classes)
        self.model = pipeline
        return self.model

    def predict(self, text):
        return OrderedDict(sorted(zip(self.model.classes_.tolist(),
                                      self.model.predict_proba([text]).tolist()[0]),
                                  key=lambda x: x[1], reverse=True))

    def test(self, text="language"):
        print("*" * 20 + " Test Starting " + "*" * 20)
        print("Testing on string: {}".format("\"" + text + "\""))

        self.build_training_set()
        print("Number of courses in training set: {}".format(len(self.training_set)))

        self.train_model()
        prediction = self.predict(text)
        pretty_prediction = [(tup[0], "{:.1%}".format(tup[1]),) for tup in list(prediction.items())[:5]]
        # print(self.model.get_params(deep=False))
        print(tabulate(pretty_prediction, headers=['Department', 'Classification Likelihood']))
        print("*" * 20 + " Test Complete " + "*" * 20)


if __name__ == "__main__":
    Classify().test()
    Classify().test(text="freedom")
    Classify().test(text="science")
    Classify().test(text="death")
    Classify().test(text="taxes")
    Classify().test(text="Twitter")
