# Spam-Filter
Spam Filter using Naive Bayes with Additive Smoothing

# Dataset
Implemend Email Spam Filter using a subset of 2005 TREC Public Spam Corpus. It contains a training set and a test set. Both files use the same format: each line represents the space-delimited properties of an email, with the first one being the email ID, the second one being whether it is a spam or ham (non-spam), and the rest are words and their occurrence numbers in this email. In preprocessing, non-word characters have been removed, and features selected similar to what Mehran Sahami did in his original paper(ftp://ftp.research.microsoft.com/pub/ejh/junkfilter.pdf) using Naive Bayes to classify spams.

