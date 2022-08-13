import class_mlp
from data import X_train, Y_train, X_test

model = class_mlp.MLP()

model.add(class_mlp.layers.layer(8, 'ReLU'))
model.add(class_mlp.layers.layer(4, 'ReLU'))
model.add(class_mlp.layers.layer(1, 'sigmoid'))

model.set_learning_rate(0.8)

model.Fit(X_train, Y_train)
acc_val = model.get_accuracy()
print(acc_val)
acc_avg_val = model.get_avg_accuracy()
print(acc_avg_val)
predictions = model.predict(X_test)
print(predictions)
