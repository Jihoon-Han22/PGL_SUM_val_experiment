# -*- coding: utf-8 -*-
import numpy as np


def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    """
    # max_len = max(len(predicted_summary), len(user_summary))
    # S = np.zeros(max_len, dtype=int)
    # G = np.zeros(max_len, dtype=int)
    # S[:len(predicted_summary)] = predicted_summary

    # G[:len(user_summary)] = user_summary
    # overlapped = S & G
    # # Compute precision, recall, f-score
    # #! for safe division
    # if sum(S) == 0:
    #     precision = 0
    # else:
    #     precision = sum(overlapped)/sum(S)
        
    # #! for safe division
    # if sum(G) == 0:
    #     recall = 0
    # else:
    #     recall = sum(overlapped)/sum(G)
    
    # if precision+recall == 0:
    #     f_score = 0
    # else:
    #     f_score = 2 * precision * recall * 100 / (precision + recall)
    
    # return f_score

    # evaluation_summary for summe, tvsum datasets
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S+1e-8)
        recall = sum(overlapped)/sum(G+1e-8)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)