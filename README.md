# Election Tweet Classifier

For details about the project, please reference `Final Report.pdf`.

---

This project includes classifiers of tweets crawled by Twitter Developer API with Naive Bayes classifier, logistic regression classifier, and
neural net classifier to analyze 2020 US presidential election. The classifiers shows how frequently supporters of each candidate use words
in their tweets.

---

1. How to run Crawler

   To run a crawler for tweets, you need tokens and secret keys given by Twitter Developer API. After you get them, create `.env` file on the root
   directory of the project and write them based on `.template.env`. After then, you can run `crawler/crawler.py` and the crawler will start.

---

2. How to run Naive Bayes Classifier

   First, you need to specify a directory where tweet data exists, which can be done in `naive-bayes/naive-bayes.py` by changing `DATASET_DIRECTORY`.
   After then, execute `naive-bayes/naive-bayes.py` on `naive-bayes/` directory.

---

3. How to run Logistic Regression Classifier, Neural Net Classifier

   Both classifiers are written as .ipynb format, which can be executed in Google Colab.
