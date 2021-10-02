# Restaurant_Review_Analysis_NLP

In this, we have to analyse the sentiment of customers towards a Restaurant.
Firstly we have to do DataPreProcessing, for that I have firstly removed all the punctuations from the review part of the datasegt, then used PotterStemmer under nltk for Stemming.
Then used CountVectorizer provided by sklearn which convert a collection of text documents to a matrix of token counts. 
Then separated the X and y from the dataset and used testtrainsplit provided by sklearn for dividing the dataset into test and train part.
Then used different models like NaiveBayes model, Neural Network model, Logistic Regression Model, KNN model and TreeClassifier model and got the best accuracy by Logisitic Regression Model.

**NaiveBayes model**

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = model.predict(X_test)     //61 percent

**Neural Network model**

model = Sequential()
model.add(Dense(input_shape=(1500,), units = 64, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 64)                96064     
_________________________________________________________________
dense_14 (Dense)             (None, 32)                2080      
_________________________________________________________________
dense_15 (Dense)             (None, 8)                 264       
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 9         
=================================================================
Total params: 98,417
Trainable params: 98,417
Non-trainable params: 0

model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.000001), metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 1000)
y_pred_nn = model.predict(X_test)
y_pred_nn
for i in range(len(y_pred_nn)):
    if(i>=0.5):
        y_pred_nn[i] = 1
    else:
        y_pred_nn[i] = 0
        
y_pred_nn 
accuracy_score(y_test, y_pred_nn)  //47 percent

**KNN**

classifier= KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2 )  
classifier.fit(X_train, y_train)  
y_pred_knn = classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)  //61 percent


**Logistic Regression**

classifier= LogisticRegression(random_state=49)  
classifier.fit(X_train, y_train) 
y_pred_log = classifier.predict(X_test)
accuracy_score(y_test,y_pred_log)  //74 percent


**Decision tree classifier**

classifier= DecisionTreeClassifier(criterion='entropy', random_state=49)  
classifier.fit(X_train, y_train)  
y_pred_tree = classifier.predict(X_test)
accuracy_score(y_test, y_pred_tree)   //68 percent
