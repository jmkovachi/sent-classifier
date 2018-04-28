

class TestResults:

    def test_input(self, decisions, classifications):
        index = 0
        
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for decision in decisions:
            if decision == classifications[index] and decision == 'positive':
                true_pos += 1
            elif decision != classifications[index] and decision == 'positive':
                false_pos += 1
            elif decision == classifications[index] and decision == 'negative':
                true_neg += 1
            elif decision != classifications[index] and decision == 'negative':
                false_neg += 1
        
            index += 1

        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_neg + false_pos)
        f_score = 2 * ((precision * recall)/(precision + recall))


        return precision, recall, accuracy, f_score