import argparse
import csv
import numpy

def spam_classifier_with_smoothing(train_file, test_file, output_file):
    # training the model
    zero_prob = 0
    additive_parameter = 18
    threshold = 5000
    file = open(train_file, 'r')
    # we will create 2 dictionaries to store wrds and their frequencies in spam and
    # non spam emails
    not_spam_word_count = dict()
    spam_word_count = dict()
    spam_count = 0
    non_spam_count = 0
    for line in file.readlines():

        count = 0
        not_spam_flag = False
        word_flag = False

        for word in line.split(" "):

            if count == 0:
                count += 1
                continue
            if count == 1:
                if word == 'ham':
                    non_spam_count += 1
                    not_spam_flag = True
                else:
                    spam_count += 1
                count += 1
                continue
            if word_flag == False:
                word_flag = True
                old_word = word
            else:
                word_flag = False
                # if the word is a number, we ignore it for classification
                if old_word.isdigit():
                    continue
                # if the length of the word is less than 4, we ignore it for classification
                if len(old_word) < 4:
                    continue
                # if the word is a stopword, we ignore it for classification
                if old_word in stop_words:
                    continue
                # if the frequency of the word in an email is greater than threshold
                # we ignore it, considering it as an outlier
                if not_spam_flag:
                    try:
                        if int(word) < threshold:
                            not_spam_word_count[old_word] += int(word)
                    except KeyError:
                        if int(word) < threshold:
                            not_spam_word_count[old_word] = int(word)
                else:
                    try:
                        if int(word) < threshold:
                            spam_word_count[old_word] += int(word)
                    except KeyError:
                        if int(word) < threshold:
                            spam_word_count[old_word] = int(word)


    spam_word_count_new = dict()

    # calculate mean and standard deviation of the spam and non-spam words' count
    mean_spam = numpy.mean(spam_word_count.values())
    mean_not_spam = numpy.mean(not_spam_word_count.values())

    std_spam = numpy.std(spam_word_count.values())
    std_not_spam = numpy.std(not_spam_word_count.values())

    # if the frequency of the word is within 1.2 standard deviations of the mean
    # then consider it for the classification, else ignore it
    for key in spam_word_count.keys():
        if spam_word_count[key] < (mean_spam + std_spam * 1.2):
            spam_word_count_new[key] = spam_word_count[key]
    not_spam_word_count_new = dict()
    for key in not_spam_word_count.keys():
        if not_spam_word_count[key] < (mean_not_spam + std_not_spam * 1.2):
            not_spam_word_count_new[key] = not_spam_word_count[key]

    # testing data
    file = open(test_file, 'r')
    count = 0
    ham_count = 0
    for line in file.readlines():

        if line.split(" ")[1] == 'ham':
            ham_count += 1
        count += 1

    line_count = 0

    file = open(test_file, 'r')
    csvfile = open(output_file, 'wb')
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)


    accuracy = 0
    count_test = 0
    probability_spam = spam_count / float(spam_count + non_spam_count)
    for line in file.readlines():

        line_count += 1
        count = 0
        probs = []
        id = ""
        for word in line.split(" "):

            if count == 0:
                id = word
                count += 1
                continue
            if count == 1:
                y_orig = word
                count += 1
                continue
            if count == 2:
                old_word_flag = False
                old_word = word
                count += 1
                continue
            if old_word_flag == False:
                if count == 3:
                    count += 1
                    try:
                        if old_word.isdigit():
                            old_word_flag = True
                            continue
                        if len(old_word) < 4:
                            old_word_flag = True
                            continue
                        if old_word in stop_words:
                            old_word_flag = True
                            continue


                        # calculating probability of individual words using the formula
                        # Pr(S/word) = (Pr(word/S) * Pr(S)) / (Pr(word/S) * Pr(S) + Pr(word/H) * Pr(H))
                        spam_prob = ((probability_spam * spam_word_count_new[old_word]) + additive_parameter) / float(
                            (probability_spam * spam_word_count_new[old_word]) + (
                                (1 - probability_spam) * not_spam_word_count_new[old_word] + additive_parameter))
                        not_spam_prob = (not_spam_word_count[old_word] + additive_parameter) / float(
                            spam_word_count[old_word] + not_spam_word_count[old_word] + additive_parameter)

                        # append the probability of each word being a spam into a list
                        # if the probability of a word being a spam is between 0.48 and 0.52
                        # then we ignore it for classification as the word is neutral and
                        # doesn't contribute towards the spamliness of the email
                        if spam_prob < 0.48 or spam_prob > 0.52:
                            probs.append(spam_prob)

                    except:
                        old_word_flag = True

                    old_word_flag = True
                    continue
                try:
                    if old_word.isdigit():
                        old_word_flag = True
                        continue
                    if len(old_word) < 4:
                        old_word_flag = True
                        continue
                    if old_word in stop_words:
                        old_word_flag = True
                        continue

                    spam_prob = ((probability_spam * spam_word_count_new[old_word]) + additive_parameter) / float(
                        (probability_spam * spam_word_count_new[old_word]) + (
                            (1 - probability_spam) * not_spam_word_count_new[old_word]) + additive_parameter)
                    not_spam_prob = (not_spam_word_count[old_word] + additive_parameter) / float(
                        spam_word_count[old_word] + not_spam_word_count[old_word] + additive_parameter)


                    if spam_prob < 0.48 or spam_prob > 0.52:
                        probs.append(spam_prob)

                except:
                    old_word_flag = True

                old_word_flag = True
            else:
                old_word_flag = False
                old_word = word
        numerator = 1.0
        denominator = 1.0

        for prob in probs:

            if prob == 0:
                zero_prob += 1
                continue

            # merging the probabilities of all the words in an email to get the probability
            # of an email being a spam using the formula
            # Pr(S/email) = (Pr(S/word 1)*Pr(S/word 2)*...*Pr(S/word n))/((Pr(S/word 1)*Pr(S/word 2)*...*Pr(S/word n)) + (Pr(H/word 1)*Pr(H/word 2)*...*Pr(H/word n)))
            numerator = numerator * prob
            denominator = denominator * (1 - prob)

        try:
            # overall probability for an email
            overall_prob = numerator / float(numerator + denominator)
        except:

            continue

        if overall_prob > 0.5:
            # the email is a spam
            spamwriter.writerow([id, 'spam'])
            count_test += 1
            if y_orig == 'spam':
                accuracy += 1

        else:
            # the email is not a spam
            spamwriter.writerow([id, 'ham'])
            if y_orig == 'ham':
                accuracy += 1

    # calculate the fina accuracy
    print "accuracy = ", accuracy / float(line_count) * 100


def spam_classifier(train_file, test_file, output_file):
    # training the model
    file = open(train_file, 'r')
    not_spam_word_count = dict()
    spam_word_count = dict()
    spam_count = 0
    non_spam_count = 0
    for line in file.readlines():

        count = 0
        not_spam_flag = False
        word_flag = False

        for word in line.split(" "):

            if count == 0:
                count += 1
                continue
            if count == 1:
                if word == 'ham':
                    non_spam_count += 1
                    not_spam_flag = True
                else:
                    spam_count += 1
                count += 1
                continue
            if word_flag == False:
                word_flag = True
                old_word = word
            else:
                word_flag = False
                if old_word.isdigit():
                    continue
                if len(old_word) < 4:
                    continue
                if old_word in stop_words:
                    continue
                if not_spam_flag:
                    try:
                        if int(word) < 5000:
                            not_spam_word_count[old_word] += int(word)
                    except KeyError:
                        if int(word) < 5000:
                            not_spam_word_count[old_word] = int(word)
                else:
                    try:
                        if int(word) < 5000:
                            spam_word_count[old_word] += int(word)
                    except KeyError:
                        if int(word) < 5000:
                            spam_word_count[old_word] = int(word)


    spam_word_count_new = dict()

    for key in spam_word_count.keys():
        if spam_word_count[key] < 5000:
            spam_word_count_new[key] = spam_word_count[key]
    not_spam_word_count_new = dict()
    for key in not_spam_word_count.keys():
        if not_spam_word_count[key] < 5000:
            not_spam_word_count_new[key] = not_spam_word_count[key]

    # testing data
    file = open(test_file, 'r')
    accuracy = 0
    count = 0
    ham_count = 0
    for line in file.readlines():

        if line.split(" ")[1] == 'ham':
            ham_count += 1
        count += 1

    line_count = 0

    file = open(test_file, 'r')
    csvfile = open(output_file, 'wb')
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    accuracy = 0
    spam_prob = 0.0
    non_spam_prob = 0.0
    count_test = 0
    probability_spam = spam_count / float(spam_count + non_spam_count)
    for line in file.readlines():

        line_count += 1
        count = 0
        probs = []
        id = ""
        for word in line.split(" "):

            if count == 0:
                id = word
                count += 1
                continue
            if count == 1:
                y_orig = word
                count += 1
                continue
            if count == 2:
                old_word_flag = False
                old_word = word
                count += 1
                continue
            if old_word_flag == False:
                if count == 3:
                    count += 1
                    try:
                        if old_word.isdigit():
                            old_word_flag = True
                            continue
                        if len(old_word) < 4:
                            old_word_flag = True
                            continue
                        if old_word in stop_words:
                            old_word_flag = True
                            continue

                        spam_prob = (probability_spam * spam_word_count_new[old_word]) / float(
                            (probability_spam * spam_word_count_new[old_word]) + (
                                (1 - probability_spam) * not_spam_word_count_new[old_word]))
                        not_spam_prob = not_spam_word_count[old_word] / float(
                            spam_word_count[old_word] + not_spam_word_count[old_word])

                        probs.append(spam_prob)

                    except:
                        old_word_flag = True

                    old_word_flag = True
                    continue
                try:
                    if old_word.isdigit():
                        old_word_flag = True
                        continue
                    if len(old_word) < 4:
                        old_word_flag = True
                        continue
                    if old_word in stop_words:
                        old_word_flag = True
                        continue

                    spam_prob = (probability_spam * spam_word_count_new[old_word]) / float(
                        (probability_spam * spam_word_count_new[old_word]) + (
                            (1 - probability_spam) * not_spam_word_count_new[old_word]))
                    not_spam_prob = not_spam_word_count[old_word] / float(
                        spam_word_count[old_word] + not_spam_word_count[old_word])

                    probs.append(spam_prob)

                except:
                    old_word_flag = True

                old_word_flag = True
            else:
                old_word_flag = False
                old_word = word
        numerator = 1.0
        denominator = 1.0

        for prob in probs:

            if prob == 0:
                continue
            numerator = numerator * prob
            denominator = denominator * (1 - prob)

        try:
            overall_prob = numerator / float(numerator + denominator)
        except:

            continue

        if overall_prob > 0.5:
            # print id, "spam"
            spamwriter.writerow([id, 'spam'])
            count_test += 1
            if y_orig == 'spam':
                accuracy += 1

        else:
            # print id, "ham"
            spamwriter.writerow([id, 'ham'])
            if y_orig == 'ham':
                accuracy += 1


    print "accuracy = ", accuracy / float(line_count) * 100


parser = argparse.ArgumentParser()
parser.add_argument('-f1', help='train dataset name', required=True)
parser.add_argument('-f2', help='test dataset name', required=True)
parser.add_argument('-o', help='output file name', required=True)

args = vars(parser.parse_args())

train_file = args['f1']
test_file = args['f2']
output_file = args['o']

stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
              'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
              'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down',
              'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
              'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself',
              'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
              'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
              'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
              'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
              'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',
              'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
              'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",
              'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
              "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your',
              'yours', 'yourself', 'yourselves']

# spam_classifier(train_file, test_file, output_file)
spam_classifier_with_smoothing(train_file, test_file, output_file)
