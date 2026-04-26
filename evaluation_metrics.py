class EvaluationMetrics:

    def confusion_matrix(self, y_true, y_pred):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
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

    def precision(self, y_true, y_pred):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred)

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self, y_true, y_pred):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def multiclass_confusion_matrix(self, y_true, y_pred):
        labels = sorted(list(set(y_true + y_pred)))
        matrix = {}

        for actual in labels:
            matrix[actual] = {}
            for predicted in labels:
                matrix[actual][predicted] = 0

        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1

        return matrix

    def precision_for_class(self, y_true, y_pred, target_class):
        tp = 0
        fp = 0

        for i in range(len(y_true)):
            if y_pred[i] == target_class and y_true[i] == target_class:
                tp += 1
            elif y_pred[i] == target_class and y_true[i] != target_class:
                fp += 1

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall_for_class(self, y_true, y_pred, target_class):
        tp = 0
        fn = 0

        for i in range(len(y_true)):
            if y_true[i] == target_class and y_pred[i] == target_class:
                tp += 1
            elif y_true[i] == target_class and y_pred[i] != target_class:
                fn += 1

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_for_class(self, y_true, y_pred, target_class):
        p = self.precision_for_class(y_true, y_pred, target_class)
        r = self.recall_for_class(y_true, y_pred, target_class)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)


metrics = EvaluationMetrics()


def print_binary_test(name, y_true, y_pred):
    tp, fp, fn, tn = metrics.confusion_matrix(y_true, y_pred)

    print(name)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print("Accuracy:", metrics.accuracy(y_true, y_pred))
    print("Precision:", metrics.precision(y_true, y_pred))
    print("Recall:", metrics.recall(y_true, y_pred))
    print("F1 Score:", metrics.f1_score(y_true, y_pred))
    print()


def print_multiclass_test(name, y_true, y_pred):
    print(name)
    print("Confusion Matrix:", metrics.multiclass_confusion_matrix(y_true, y_pred))
    print("Accuracy:", metrics.accuracy(y_true, y_pred))

    labels = sorted(list(set(y_true + y_pred)))
    for label in labels:
        print("Class", label, "Precision:", metrics.precision_for_class(y_true, y_pred, label))
        print("Class", label, "Recall:", metrics.recall_for_class(y_true, y_pred, label))
        print("Class", label, "F1:", metrics.f1_for_class(y_true, y_pred, label))

    print()


# Required test cases
print_binary_test("Test Case 1 - Perfect Classification",
                  [1, 0, 1, 0],
                  [1, 0, 1, 0])

print_binary_test("Test Case 2 - All Wrong",
                  [1, 1, 0, 0],
                  [0, 0, 1, 1])

print_binary_test("Test Case 3 - No Predicted Positives",
                  [1, 0, 1, 0],
                  [0, 0, 0, 0])

print_binary_test("Test Case 4 - No Actual Positives",
                  [0, 0, 0, 0],
                  [1, 0, 1, 0])


# Additional test cases
print_binary_test("High Recall, Low Precision",
                  [1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1])

print_binary_test("High Precision, Low Recall",
                  [1, 1, 1, 1, 0, 0],
                  [1, 0, 0, 0, 0, 0])

print_binary_test("Rare Positive Case",
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# Multi-class test cases
print_multiclass_test("Multi-class Test 1",
                      [0, 1, 2, 1, 0, 2],
                      [0, 2, 2, 1, 0, 1])

print_multiclass_test("Multi-class Test 2 - Perfect Classification",
                      [0, 1, 2, 2, 1, 0],
                      [0, 1, 2, 2, 1, 0])