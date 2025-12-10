from mysklearn.myclassifiers import MyDecisionTreeClassifier
# decision tree tests
def test_decision_tree_classifier_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True",
                         "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
        ["Attribute", "att0",
         ["Value", "Junior",
          ["Attribute", "att3",
           ["Value", "no",
            ["Leaf", "True", 3, 5]
            ],
           ["Value", "yes",
            ["Leaf", "False", 2, 5]
            ]
           ]
          ],
         ["Value", "Mid",
          ["Leaf", "True", 4, 14]
          ],
         ["Value", "Senior",
          ["Attribute", "att2",
           ["Value", "no",
            ["Leaf", "False", 3, 5]
            ],
           ["Value", "yes",
            ["Leaf", "True", 2, 5]
            ]
           ]
          ]
         ]

    # iphone data
    X_train_iphone = [
        [1, 3, "fair"], [1, 3, "excellent"], [2, 3, "fair"],
        [2, 2, "fair"], [2, 1, "fair"], [2, 1, "excellent"],
        [2, 1, "excellent"], [1, 2, "fair"], [1, 1, "fair"],
        [2, 2, "fair"], [1, 2, "excellent"], [2, 2, "excellent"],
        [2, 3, "fair"], [2, 2, "excellent"], [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes",
                      "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone = [
        "Attribute", "att0",
        ["Value", 1,
         ["Attribute", "att1",
          ["Value", 1,
           ["Leaf", "yes", 1, 15]
           ],
          ["Value", 2,
           ["Attribute", "att2",
            ["Value", "excellent",
             ["Leaf", "yes", 1, 15]
             ],
            ["Value", "fair",
             ["Leaf", "no", 1, 15]
             ]
            ]
           ],
          ["Value", 3,
           ["Leaf", "no", 2, 15]
           ]
          ]
         ],
        ["Value", 2,
         ["Attribute", "att2",
          ["Value", "excellent",
           ["Leaf", "no", 4, 15]
           ],
          ["Value", "fair",
           ["Leaf", "yes", 6, 15]
           ]
          ]
         ]
    ]

    # interview classification
    interview_classifier = MyDecisionTreeClassifier()
    interview_classifier.fit(X_train_interview, y_train_interview)
    assert interview_classifier.tree == tree_interview

    # iphone classification
    iphone_classifier = MyDecisionTreeClassifier()
    iphone_classifier.fit(X_train_iphone, y_train_iphone)
    assert iphone_classifier.tree == tree_iphone


def test_decision_tree_classifier_predict():
    # interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True",
                         "True", "True", "False"]

    # Test instances from B Attribute Selection Lab Task #2
    X_test_interview = [
        ["Junior", "Java", "yes", "no"],
        ["Junior", "Java", "yes", "yes"]
    ]
    y_expected_interview = ["True", "False"]

    # Train and predict
    interview_classifier = MyDecisionTreeClassifier()
    interview_classifier.fit(X_train_interview, y_train_interview)
    y_pred_interview = interview_classifier.predict(X_test_interview)

    assert y_pred_interview == y_expected_interview

    # iphone dataset
    X_train_iphone = [
        [1, 3, "fair"], [1, 3, "excellent"], [2, 3, "fair"],
        [2, 2, "fair"], [2, 1, "fair"], [2, 1, "excellent"],
        [2, 1, "excellent"], [1, 2, "fair"], [1, 1, "fair"],
        [2, 2, "fair"], [1, 2, "excellent"], [2, 2, "excellent"],
        [2, 3, "fair"], [2, 2, "excellent"], [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes",
                      "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # Test instances
    X_test_iphone = [
        [2, 2, "fair"],  # standing=2, job=2, credit=fair -> yes
        [1, 1, "excellent"]  # standing=1, job=1, credit=excellent -> yes
    ]
    y_expected_iphone = ["yes", "yes"]

    # Train and predict
    iphone_classifier = MyDecisionTreeClassifier()
    iphone_classifier.fit(X_train_iphone, y_train_iphone)
    y_pred_iphone = iphone_classifier.predict(X_test_iphone)

    assert y_pred_iphone == y_expected_iphone