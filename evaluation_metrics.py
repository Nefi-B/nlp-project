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
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred)
        total = tp + fp + fn + tn

        if total == 0:
            return 0.0

        return (tp + tn) / total


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


metrics = EvaluationMetrics()


def print_test(name, y_true, y_pred):
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


# Required test cases
print_test("Test Case 1", [1, 0, 1, 0], [1, 0, 1, 0])
print_test("Test Case 2", [1, 1, 0, 0], [0, 0, 1, 1])
print_test("Test Case 3", [1, 0, 1, 0], [0, 0, 0, 0])
print_test("Test Case 4", [0, 0, 0, 0], [1, 0, 1, 0])


# Additional test cases
print_test("High Recall, Low Precision",
           [1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1])

print_test("High Precision, Low Recall",
           [1, 1, 1, 1, 0, 0],
           [1, 0, 0, 0, 0, 0])

print_test("Rare Positive Case",
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])