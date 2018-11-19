from sklearn import svm


X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC(gamma='scale')


clf.fit(X, y)

print("predict=", clf.predict([[2., 2.]]))