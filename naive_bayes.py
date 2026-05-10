import csv
import html
import math
import random
import re
import sys
import zipfile


class EvaluationMetrics:
    def confusion_matrix(self, y_true, y_pred, positive_label="positive"):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(len(y_true)):
            if y_true[i] == positive_label and y_pred[i] == positive_label:
                tp += 1
            elif y_true[i] != positive_label and y_pred[i] == positive_label:
                fp += 1
            elif y_true[i] == positive_label and y_pred[i] != positive_label:
                fn += 1
            else:
                tn += 1

        return tp, fp, fn, tn

    def accuracy(self, y_true, y_pred):
        if len(y_true) == 0:
            return 0.0

        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1

        return correct / len(y_true)

    def precision(self, y_true, y_pred, positive_label="positive"):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred, positive_label)

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self, y_true, y_pred, positive_label="positive"):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred, positive_label)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_score(self, y_true, y_pred, positive_label="positive"):
        p = self.precision(y_true, y_pred, positive_label)
        r = self.recall(y_true, y_pred, positive_label)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)


class IMDBNaiveBayesClassifier:
    def __init__(self):
        self.class_counts = {}
        self.word_counts = {}
        self.total_words = {}
        self.vocabulary = set()
        self.labels = []
        self.total_documents = 0

        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "while", "of", "to",
            "in", "on", "for", "with", "as", "at", "by", "from", "this",
            "that", "these", "those", "is", "are", "was", "were", "be",
            "been", "being", "it", "its", "i", "you", "he", "she", "we",
            "they", "them", "his", "her", "their", "our", "my", "your",
            "me", "him", "so", "very", "too", "just", "about", "into",
            "than", "then", "there", "here", "when", "where", "what",
            "which", "who", "whom", "why", "how"
        }

    def clean_text(self, text):
        text = html.unescape(text)
        text = re.sub(r"<.*?>", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text):
        text = self.clean_text(text)
        words = text.split()

        tokens = []
        for word in words:
            if len(word) < 2:
                continue
            if word in self.stop_words:
                continue
            tokens.append(word)

        return tokens

    def fit(self, texts, labels):
        self.class_counts = {}
        self.word_counts = {}
        self.total_words = {}
        self.vocabulary = set()
        self.labels = []
        self.total_documents = len(texts)

        for i in range(len(texts)):
            label = labels[i]

            if label not in self.class_counts:
                self.class_counts[label] = 0
                self.word_counts[label] = {}
                self.total_words[label] = 0
                self.labels.append(label)

            self.class_counts[label] += 1

            tokens = self.tokenize(texts[i])

            for token in tokens:
                self.vocabulary.add(token)

                if token not in self.word_counts[label]:
                    self.word_counts[label][token] = 0

                self.word_counts[label][token] += 1
                self.total_words[label] += 1

    def predict_one(self, text):
        tokens = self.tokenize(text)
        vocabulary_size = len(self.vocabulary)

        best_label = None
        best_score = None

        for label in self.labels:
            class_probability = self.class_counts[label] / self.total_documents
            score = math.log(class_probability)

            denominator = self.total_words[label] + vocabulary_size

            for token in tokens:
                token_count = 0

                if token in self.word_counts[label]:
                    token_count = self.word_counts[label][token]

                token_probability = (token_count + 1) / denominator
                score += math.log(token_probability)

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict(self, texts):
        predictions = []

        for text in texts:
            predictions.append(self.predict_one(text))

        return predictions


def load_imdb_dataset(path):
    reviews = []
    labels = []

    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zip_file:
            csv_name = None

            for name in zip_file.namelist():
                if name.endswith(".csv"):
                    csv_name = name
                    break

            if csv_name is None:
                raise ValueError("No CSV file found inside the ZIP file.")

            with zip_file.open(csv_name, "r") as file:
                text_file = (line.decode("utf-8", errors="replace") for line in file)
                reader = csv.DictReader(text_file)

                for row in reader:
                    reviews.append(row["review"])
                    labels.append(row["sentiment"])
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            reader = csv.DictReader(file)

            for row in reader:
                reviews.append(row["review"])
                labels.append(row["sentiment"])

    return reviews, labels


def train_test_split(texts, labels, test_size=0.2, seed=42):
    indices = list(range(len(texts)))
    random.seed(seed)
    random.shuffle(indices)

    test_count = int(len(texts) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for index in train_indices:
        x_train.append(texts[index])
        y_train.append(labels[index])

    for index in test_indices:
        x_test.append(texts[index])
        y_test.append(labels[index])

    return x_train, x_test, y_train, y_test


def run_evaluation(y_true, y_pred):
    metrics = EvaluationMetrics()

    tp, fp, fn, tn = metrics.confusion_matrix(
        y_true,
        y_pred,
        positive_label="positive"
    )

    print("Confusion Matrix")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print()

    print("Accuracy:", round(metrics.accuracy(y_true, y_pred), 4))
    print("Precision:", round(metrics.precision(y_true, y_pred, "positive"), 4))
    print("Recall:", round(metrics.recall(y_true, y_pred, "positive"), 4))
    print("F1 Score:", round(metrics.f1_score(y_true, y_pred, "positive"), 4))


def run_metric_tests():
    metrics = EvaluationMetrics()

    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "positive", "negative"]

    assert metrics.confusion_matrix(y_true, y_pred, "positive") == (2, 0, 0, 2)
    assert metrics.accuracy(y_true, y_pred) == 1.0
    assert metrics.precision(y_true, y_pred, "positive") == 1.0
    assert metrics.recall(y_true, y_pred, "positive") == 1.0
    assert metrics.f1_score(y_true, y_pred, "positive") == 1.0

    y_true = ["positive", "positive", "negative", "negative"]
    y_pred = ["negative", "negative", "positive", "positive"]

    assert metrics.confusion_matrix(y_true, y_pred, "positive") == (0, 2, 2, 0)
    assert metrics.accuracy(y_true, y_pred) == 0.0
    assert metrics.precision(y_true, y_pred, "positive") == 0.0
    assert metrics.recall(y_true, y_pred, "positive") == 0.0
    assert metrics.f1_score(y_true, y_pred, "positive") == 0.0

    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["negative", "negative", "negative", "negative"]

    assert metrics.confusion_matrix(y_true, y_pred, "positive") == (0, 0, 2, 2)
    assert metrics.precision(y_true, y_pred, "positive") == 0.0
    assert metrics.recall(y_true, y_pred, "positive") == 0.0
    assert metrics.f1_score(y_true, y_pred, "positive") == 0.0

    print("All metric tests passed.")
    print()


def main():
    csv.field_size_limit(sys.maxsize)

    dataset_path = "IMDB Dataset.csv"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    run_metric_tests()

    print("Loading dataset...")
    texts, labels = load_imdb_dataset(dataset_path)

    print("Total examples:", len(texts))

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        seed=42
    )

    print("Training examples:", len(x_train))
    print("Test examples:", len(x_test))
    print()

    classifier = IMDBNaiveBayesClassifier()

    print("Training classifier...")
    classifier.fit(x_train, y_train)

    print("Predicting test set...")
    predictions = classifier.predict(x_test)

    print()
    run_evaluation(y_test, predictions)


if __name__ == "__main__":
    main()