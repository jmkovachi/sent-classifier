import classifyPOS
import Reuters_PMI

class TestResults:

    @staticmethod
    def test_input(classifications, decisions):
        index = 0
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for classification in classifications:
            if classification == decisions[index] and classification == 'positive':
                true_pos += 1
            elif classification != decisions[index] and classification == 'positive':
                false_pos += 1
            elif classification == decisions[index] and classification == 'negative':
                true_neg += 1
            elif classification != decisions[index] and classification == 'negative':
                false_neg += 1
        
            index += 1

        return TestResults.compute_scores(true_pos, true_neg, false_pos, false_neg)

    @staticmethod
    def compute_scores(true_pos, true_neg, false_pos, false_neg):
        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_neg + false_pos)
        f_score = 2 * ((precision * recall)/(precision + recall))

        return precision, recall, accuracy, f_score

    @staticmethod
    def print_scores(precision, recall, accuracy, f_score, method):
        print(' Type: {} \n Precision: {} \n Recall: {} \n Accuracy: {} \n F score: {} \n'.format(method, precision, recall, accuracy, f_score))

    @staticmethod 
    def test_all_methods():
        nb_trainer = classifyPOS.NB_Trainer()
        svm_trainer = classifyPOS.SVM_Trainer()
        lexicon_test = Reuters_PMI.McDonald_Word_List()

        '''
        try:
            print('Beginning Lexicon testing..')
            precision, recall, accuracy, f_score = lexicon_test.compute_lexicon_score()
            TestResults.print_scores('lexicon', precision, recall, accuracy, f_score)
        except Exception as e:
            print('Lexicon failed : ' + str(e))
        '''

        try:
            print('Beginning SVM testing..')
            svm_test_set = svm_trainer.train(train_titles=True)
            precision, recall, accuracy, f_score = svm_trainer.test(svm_test_set, test)
            TestResults.print_scores(precision, recall, accuracy, f_score, 'SVM')
        except Exception as e:
            print('SVM failed: ' + str(e))

        try:
            print('Beginning NB testing..')
            nb_trainer.nltk_train_semeval()
            precision, recall, accuracy, f_score = nb_trainer.test(test_titles=True)
            TestResults.print_scores(precision, recall, accuracy, f_score, 'NB')
        except Exception as e:
            print('NB failed: ' + str(e))


if __name__ == "__main__":
    TestResults.test_all_methods()
    