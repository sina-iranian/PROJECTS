2(TDIDF and NaiveBeues of the dataset of 1(user_courses_review_09_2023)).ipynb
Overview

Builds a text-classification pipeline to predict positive (≥ 4 stars) vs. not-positive course reviews using TF-IDF features and a Multinomial Naive Bayes classifier, with SMOTE to handle class imbalance. Includes performance reports and a ranked list of the most influential words per class.

Pipeline Steps
1) Load Data

Read CSV → df

Rename columns: course_name, lecture_name, review_rating, review_comment

2) Clean & Label

Drop rows where both review_rating and review_comment are missing

Fill missing review_comment with ''

Coerce review_rating to numeric; drop non-numeric rows

Create binary target label: 1 if rating ≥ 4, else 0

3) Text Preprocessing

Download NLTK assets: stopwords, wordnet

clean_text():

lowercase

remove punctuation & digits

remove English stopwords

lemmatize tokens

Output column: cleaned_comment

4) Vectorization (TF-IDF)

TfidfVectorizer(max_features=5000) → features X

Target y = label

5) Train/Test Split

train_test_split(test_size=0.2, random_state=42)

5.5) Imbalance Handling (SMOTE)

Apply SMOTE to training set only → X_train_res, y_train_res

6) Model Training

Train Multinomial Naive Bayes on resampled training data

7) Prediction & Evaluation

Predict on untouched test set

Print classification_report, plot confusion matrix, and print accuracy

8) Model Interpretation (Word Importance)

Get TF-IDF feature_names

Use model.feature_log_prob_ to rank words by class likelihood

Print all words ranked by importance for:

Class 1 (positive)

Class 0 (not-positive)

Results (this run)
Accuracy ≈ 0.857
Support: class 0 = 83, class 1 = 2082

Class 0 (not-positive): precision 0.17, recall 0.73, F1 0.28
Class 1 (positive):     precision 0.99, recall 0.86, F1 0.92


High overall accuracy driven by the dominant positive class. SMOTE improved recall for class 0, but precision remains low due to strong class imbalance.

Notes & Quick Improvements

Report class-balanced metrics (balanced accuracy, macro-F1, ROC-AUC / PR-AUC)

Tune TfidfVectorizer (e.g., ngram_range=(1,2), min_df, max_df) and MultinomialNB(alpha=...)

Consider Logistic Regression or Linear SVM for better minority precision

Revisit the label threshold (≥ 4) or move to 3-class (negative / neutral / positive)

Review domain stopwords; assess lemmatization vs. stemming trade-offs

--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
###3(IMPROVING 2, USING TDIDF AND NAIVEBEYES FOR THE EXTACTION OF ALL POSITIVE AND NEGATIVE WORDS OF THE DATASET)


Overview

Builds a text-classification pipeline to predict positive (≥ 4 stars) vs. not-positive course reviews using TF-IDF features and a Multinomial Naive Bayes classifier. It includes evaluation (report + confusion matrix) and interprets model behavior by ranking influential words per class.

Pipeline Steps

1) Load Data

Read CSV → df

Rename columns: course_name, lecture_name, review_rating, review_comment

2) Clean & Label

Drop rows where both review_rating and review_comment are missing

Fill missing review_comment with ''

Coerce review_rating → numeric; drop rows that can’t be coerced

Create binary label: 1 if rating ≥ 4, else 0

3) Text Preprocessing

NLTK downloads: stopwords, wordnet

clean_text():

lowercase

remove punctuation & digits

remove English stopwords

lemmatize tokens

Output: cleaned_comment

4) Vectorization (TF-IDF)

TfidfVectorizer(max_features=5000) → sparse features X

Target y = label

5) Train/Test Split

train_test_split(test_size=0.2, random_state=42) → X_train, X_test, y_train, y_test

6) Train Model

Multinomial Naive Bayes: model.fit(X_train, y_train)

7) Predict & Evaluate

Predict on untouched test set: y_pred = model.predict(X_test)

Print classification_report, plot confusion matrix, print accuracy

8) Interpretability (Word Importance)

Get TF-IDF feature_names

Use model.feature_log_prob_ to rank words by per-class likelihood

Print all words ranked by importance for:

Class 1 (positive)

Class 0 (not-positive)

Results (this run)
Accuracy: ≈ 0.96
Support: class 0 = 83, class 1 = 2082

Class 0 (not-positive): precision 0.00, recall 0.00, F1 0.00
Class 1 (positive):     precision 0.96, recall 1.00, F1 0.98


Very high overall accuracy due to extreme class imbalance; the model predicts almost all reviews as positive, yielding zero performance on class 0.

Notes & Quick Improvements

Report balanced metrics (balanced accuracy, macro-F1, PR-AUC) to reflect minority-class performance

Address imbalance:

Use class weights (for models that support it) or resampling (e.g., SMOTE/undersampling) on the training set only

Tune text features:

ngram_range=(1,2), min_df, max_df

Try removing or keeping certain domain stopwords

Try alternative linear models for better minority precision:

Logistic Regression, Linear SVM

Revisit the labeling scheme:

Consider 3-class (negative / neutral / positive) if it better fits the task goals



--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------



###4(Adding SMOTE method for Balancing and Resampling while using TDIDF AND NAIVEBEYES TO DO THE EXTACTION OF ALL POSITIVE AND NEGATIVE WORDS OF THE DATASET)




Overview

End-to-end text-classification pipeline to predict positive (≥ 4★) vs. not-positive course reviews using TF-IDF features and Multinomial Naive Bayes, with SMOTE to mitigate class imbalance. Includes evaluation (report + confusion matrix) and model interpretability via top words per class.

Pipeline Steps

Load Data

Read CSV → df

Rename: course_name, lecture_name, review_rating, review_comment

Clean & Label

Drop rows where both review_rating & review_comment are missing

Fill missing review_comment with ''

Coerce review_rating → numeric; drop NaNs

Create label: 1 if rating ≥ 4, else 0

Text Preprocessing

NLTK assets: stopwords, wordnet

clean_text():

lowercase

remove punctuation & digits

remove English stopwords

lemmatize

Output: cleaned_comment

Vectorization (TF-IDF)

TfidfVectorizer(max_features=5000) → X

Target y = label

Train/Test Split

train_test_split(test_size=0.2, random_state=42)

5.5) Imbalance Handling

SMOTE on training set only → X_train_res, y_train_res

Model Training

Multinomial Naive Bayes on resampled data

Predict & Evaluate

Predict on untouched test set

Print classification_report, plot confusion matrix, print accuracy

Model Interpretation

feature_names = vectorizer.get_feature_names_out()

model.feature_log_prob_ ranks words by class likelihood

Print all words ranked for:

Class 1 (positive)

Class 0 (not-positive)

Results (this run)
Accuracy: 0.8568
Support: class 0 = 83, class 1 = 2082

Class 0 (not-positive): precision 0.17, recall 0.73, F1 0.28
Class 1 (positive):     precision 0.99, recall 0.86, F1 0.92


Overall accuracy is high but driven by the dominant positive class. SMOTE boosts recall for class 0; precision remains low due to imbalance.





--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


###6(THE HYBRID APPROACH OF USING ###5 and ###6)

Overview

A hybrid text-classification pipeline that predicts positive (≥ 4★) vs. not-positive course reviews.
It combines:

CODE1-style text cleaning (lowercasing, punctuation/digit removal, stopword removal, lemmatization),

CODE2-style vectorization + imbalance handling (TF-IDF + SMOTE),

and trains a Multinomial Naive Bayes classifier.

Includes evaluation (classification report, confusion matrix) and is structured for reuse via a pipeline.

Pipeline Steps

Load & Label

Read CSV → df; rename columns.

Clean nulls; coerce review_rating to numeric.

Create binary label: 1 if rating ≥ 4, else 0.

Keep only review_comment, label.

Text Preprocessing (CODE1 Style)

Download NLTK assets: stopwords, wordnet.

clean_text():

lowercase

remove punctuation & digits

remove English stopwords

lemmatize

Output → cleaned_comment.

Split & Vectorize

train_test_split(test_size=0.2, random_state=42) on the cleaned text.

TfidfVectorizer(max_features=5000) → X_train_tfidf, X_test_tfidf.

Imbalance Handling (SMOTE)

Apply SMOTE on the training set only to balance classes: X_train_balanced, y_train_balanced.

Train

Multinomial Naive Bayes trained on the SMOTE-balanced TF-IDF features.

A make_pipeline(vectorizer, model) is also created for convenient reuse.

Evaluate

Predict on the untouched TF-IDF test set.

Report: classification_report, accuracy, and confusion matrix.

Results (this run)
Accuracy: 0.8587
Support: class 0 = 83, class 1 = 2082

Class 0 (not-positive): precision 0.18, recall 0.73, F1 0.29
Class 1 (positive):     precision 0.99, recall 0.86, F1 0.92


Takeaway: SMOTE notably boosts recall on the minority class (0), but precision remains low due to extreme imbalance. Overall accuracy is high, dominated by class 1.










--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------



###7(THE HYBRID APPROACH OF USING ###6 TEST AND TRAIN DATA)



Overview

Builds a robust text-classification pipeline to predict positive (≥ 4★) vs. not-positive course reviews using separate training and external test files.
Pipeline combines:

Clean labeling,

CODE1-style text cleaning (lowercase, punctuation/digit removal, stopword removal, lemmatization),

TF-IDF vectorization,

SMOTE on the training set only for imbalance,

Multinomial Naive Bayes classifier,

Evaluation on the external test set (no leakage).

Pipeline Steps

Load & Label (Train/Test Files)

Load 1(user_courses_review_09_2023).csv (train) and 2(user_courses_review_test_set).csv (test).

Standardize columns: course_name, lecture_name, review_rating, review_comment.

Clean nulls, coerce review_rating → numeric, drop invalids.

Create binary label: 1 if rating ≥ 4, else 0.

Keep only: review_comment, label.

Text Preprocessing (unchanged from #6)

NLTK: stopwords, wordnet.

clean_text() → lowercase, remove punctuation & digits, remove English stopwords, lemmatize.

Add cleaned_comment to both train and test.

⚠️ Note: You may see SettingWithCopyWarning. Use df = df.copy() before assigning new columns or use .loc[:, 'cleaned_comment'] = ....

Vectorize + Balance

TfidfVectorizer(max_features=5000) fit on train, transform train & test.

SMOTE on training TF-IDF features only → X_train_balanced, y_train_balanced.

Train

Train Multinomial Naive Bayes on SMOTE-balanced training vectors.

Evaluate (External Test Set)

Predict on untouched test TF-IDF.

Print accuracy, classification report, and plot confusion matrix.

Results (this run)
Accuracy: 0.85065
Support: class 0 = 15, class 1 = 139

Class 0 (not-positive): precision 0.38, recall 0.87, F1 0.53
Class 1 (positive):     precision 0.98, recall 0.85, F1 0.91

Macro avg:  precision ~0.68, recall ~0.86, F1 ~0.72
Weighted avg: precision ~0.92, recall ~0.85, F1 ~0.87


Takeaway: Using a truly external test set confirms generalization. SMOTE significantly improves recall for the minority class (0), while precision for class 0 remains modest due to imbalance and short texts.












--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------

###10(USING THE HYBRID APPROACH OF NAIVE-BEYES ABD SVM FOR THE IMPLEMENTATION OF TDIDF)


Overview

End-to-end sentiment classifier for course reviews (⭐ positive ≥ 4 vs. not-positive), trained on a separate train set and evaluated on an external test set.
The pipeline uses:

CODE-style text cleaning (lowercase, punctuation/digit removal, stopwords, lemmatization),

TF-IDF features,

SMOTE on the training set only (to mitigate class imbalance),

Two models: Multinomial Naive Bayes + Linear SVM,

Soft voting (weighted average of probabilities/scores) for final predictions,

Word-level importance analysis (NB + SVM hybrid score),

CSV/Excel export and bar-chart visualizations of positive/negative words.

Pipeline Steps

Load & Label (Train/Test Files)

Read 1(user_courses_review_09_2023).csv (train) and 2(user_courses_review_test_set).csv (test).

Standardize columns → course_name, lecture_name, review_rating, review_comment.

Clean: drop empty rows, coerce review_rating to numeric, remove invalids.

Create label: 1 if rating ≥ 4, else 0.

Keep only review_comment, label.

Text Preprocessing

NLTK: download stopwords, wordnet.

clean_text():

lowercase

remove punctuation & digits

remove English stopwords

lemmatize

Add cleaned_comment to train and test.

⚠️ If you see SettingWithCopyWarning, use df = df.copy() or .loc[:, 'cleaned_comment'] = ....

Vectorize + Balance

TfidfVectorizer(max_features=5000) fit on train, transform train & test.

SMOTE applied to training TF-IDF only → balanced X_train_balanced, y_train_balanced.

Train Two Models

Naive Bayes: MultinomialNB().fit(...)

Linear SVM: LinearSVC(max_iter=10000, random_state=42).fit(...)

Hybrid Soft Voting

NB probabilities: predict_proba on test.

SVM scores: decision_function → mapped via sigmoid to pseudo-probabilities.

Weighted average: hybrid_probs = 0.5*nb + 0.5*svm → threshold 0.5 → hybrid_preds.

💡 For calibrated SVM probabilities, consider CalibratedClassifierCV instead of manual sigmoid.

Evaluate (External Test Set)

Print accuracy, classification report.

Plot confusion matrix.

Results (this run)
Accuracy: 0.8636
Support: class 0 = 15, class 1 = 139

Class 0 (not-positive): precision 0.41, recall 0.87, F1 0.55
Class 1 (positive):     precision 0.98, recall 0.86, F1 0.92

Macro avg:    P 0.69 | R 0.86 | F1 0.74
Weighted avg: P 0.93 | R 0.86 | F1 0.88


Takeaway: The hybrid soft vote meaningfully boosts recall on the minority class (0) while keeping very high precision for the majority class.

Word Importance & Visualizations

Feature names: from TF-IDF.

NB importance: feature_log_prob_[1] - feature_log_prob_[0].

SVM importance: coef_[0] (larger → more positive).

Hybrid score: average of NB and SVM importances.

Saved hybrid_word_scores.xlsx, and plotted:

Full sentiment spectrum (negative → positive words),

Top 15 negative and top 15 positive words bar chart.

If you get an xlsxwriter version warning, update to ≥ 1.4.3 or use engine='openpyxl'.












