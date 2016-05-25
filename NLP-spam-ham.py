
from nltk import word_tokenize,WordNetLemmatizer,classify,NaiveBayesClassifier
from nltk.corpus import stopwords
import random
import glob,os
spam_mail = []
ham_mail = []

#Creating the spam list
spam_path=os.path.join('spam/', '*.txt')

#Using glob to scan through all text files in spam directory
for text_file in glob.glob(spam_path ): 
	write_file = open(text_file, "rt", encoding='latin1')
	spam_mail.append(write_file.read())
	write_file.close()

#Creating the spam list
ham_path=os.path.join('ham/', '*.txt')

#Using glob to scan through all text files in ham directory
for text_file in glob.glob(ham_path): 
	write_file = open(text_file, "rt", encoding='latin1')
	ham_mail.append(write_file.read())
	write_file.close()

#Merging and shuffling all the data for even level of training
mail_shuffle = ([(mail,'spam') for mail in spam_mail] + [(mail,'ham') for mail in
ham_mail])
random.shuffle(mail_shuffle)

#Printing the forst 5 test emails for testing the trainined machine
size_test = int(len(mail_shuffle) * 0.10)
for test_mail in mail_shuffle[:5]:
	print (test_mail)
	print ("\n")

#Assigning the stopwords and word lemmatizer functions
stpwords = stopwords.words('english')
word_limit = WordNetLemmatizer()


#Function for extracting features/raw words from the emails
def raw_mail(org_email):

	features = {}
	#Breaking the emails into words and stemming them to extact actual words
	wordtokens = [word_limit.lemmatize(key.lower( )) for key in word_tokenize(org_email)]
	for key in wordtokens:
		if key not in stpwords:
			features[key] = True
	return features


def test_raw_mail(org_email):

	features_test = {}
	wordtokens_test = [word_limit.lemmatize(key.lower()) for key in
	word_tokenize(org_email)]
	for key in wordtokens_test:
		if key not in stpwords:
			features_test[key] = True
	return features_test

	#Extracting the features(Tonenized, stemmed and non-stopwords emails) from all the emails
	feature_sets = [(raw_mail(n), g) for (n,g) in mail_shuffle]

	#Splitting the test and training data sets from the whole email set features
	size_feature = int(len(feature_sets) * 0.10)
	train_set, test_set = feature_sets[size_feature:], feature_sets[:size_feature]
	classifier = NaiveBayesClassifier.train(train_set)
	#print (test_set[1:5])

	#Printing the accuracy of the machine
	print ('accuracy of the machine: ', (classify.accuracy(classifier,test_set))*100) 
	
	#Printing the top 50 features
	classifier.show_most_informative_features(50) 

	#Printing the spam and ham labels
	print ('labels:',classifier.labels())

	#Classification of user entered email
	while(True):
		featset = raw_mail(input("Enter text to classify: "))
		print (classifier.classify(featset))