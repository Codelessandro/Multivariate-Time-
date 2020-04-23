from matplotlib import pyplot as plt
import pandas as pd



def save_score(scores,dataset, target_month, window_length, score, model):

    return scores.append({
        'model': str(model),
        'target_month': target_month,
        'window_length': window_length,
        'score': score
    }, ignore_index=True)


def best_scores(scores,model_substr):
    scores=scores[scores.model.str.contains(model_substr)]
    best_foreach_target = scores.groupby('target_month').agg({"score": min})
    return  best_foreach_target


def plot_scores(scores,model_substr):
    best_foreach_target = best_scores(scores,model_substr)
    plt.plot(best_foreach_target.index.values, best_foreach_target.score.values, label=model_substr)

    plt.xlabel("Target Month")
    plt.ylabel("Average Distance")
    plt.legend(loc='best')


def plot_scores(scores,model_substr):
    best_foreach_target = best_scores(scores,model_substr)
    plt.plot(best_foreach_target.index.values, best_foreach_target.score.values, label=model_substr)

    plt.xlabel("Target Month")
    plt.ylabel("Average Distance")
    plt.legend(loc='best')



