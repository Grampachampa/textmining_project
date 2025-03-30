import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import csv
from transformers import pipeline
import matplotlib.pyplot as plt
import time

# Reusing this function from lab3
def vader_output_to_label(vader_output):
    compound = vader_output['compound']
    if compound < 0:
        return 'negative'
    elif compound == 0.0:
        return 'neutral'
    else:
        return 'positive'

vader_model = SentimentIntensityAnalyzer()
sentences = []


with open('datasets/sentiment-topic-test.tsv') as f:
    tsv_reader = list(csv.reader(f, delimiter="\t"))
sentiment_tsv = {i[0]: {"sentence": i[1], "sentiment": i[2]} for i in tsv_reader[1:]}

start_vader = time.time()
all_vader_output = []
gold = []
for i in sentiment_tsv:
    gold.append(sentiment_tsv[i]["sentiment"])
    vader_output = vader_model.polarity_scores(sentiment_tsv[i]["sentence"])
    all_vader_output.append(vader_output_to_label(vader_output))

end_vader = time.time()
vader_time = end_vader - start_vader
report_vader = classification_report(gold, all_vader_output, output_dict=True, digits=4)

sentimentenglish = pipeline("sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")
start_roberta = time.time()
transformer_sentiment_outputs = []
for i in sentiment_tsv:
    transformer_sentiment_output = sentimentenglish(sentiment_tsv[i]["sentence"])
    transformer_sentiment_outputs.append(transformer_sentiment_output[0]["label"])
end_roberta = time.time()
roberta_time = end_roberta - start_roberta
report_transformer = classification_report(gold, transformer_sentiment_outputs, output_dict=True, digits=4)


def plot_classification_reports(report1, report2, title1="VADER", title2="RoBERTa"):
    labels = list(report1.keys())[:-3]
    metrics = ["precision", "recall", "f1-score"]
    
    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.8 + 2))
    ax.set_axis_off()
    table_data = [["Metric", "Class", title1, title2]]

    for label in labels:
        for metric in metrics:
            row = [metric, label, f"{report1[label][metric]:.4f}", f"{report2[label][metric]:.4f}"]
            table_data.append(row)

    for avg_type in ["macro avg", "weighted avg"]:
        for metric in metrics:
            row = [metric, avg_type, f"{report1[avg_type][metric]:.4f}", f"{report2[avg_type][metric]:.4f}"]
            table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title("Classification Report Comparison", fontsize=14, fontweight="bold")
    plt.savefig("classification_report_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()


plot_classification_reports(report_vader, report_transformer)

def print_sentence_comparisons():
    print("\nSentence-Level Predictions\n" + "=" * 50)
    for i, key in enumerate(sentiment_tsv):
        sentence = sentiment_tsv[key]["sentence"]
        gold_label = sentiment_tsv[key]["sentiment"]
        vader_label = all_vader_output[i]
        transformer_label = transformer_sentiment_outputs[i]
        print(key)
        print(f"Sentence: {sentence}")
        print(f"  Gold Label: {gold_label}")
        print(f"  VADER Prediction: {vader_label}")
        print(f"  RoBERTa Prediction: {transformer_label}")
        print("-" * 50)


print_sentence_comparisons()
print(f"\nExecution Time:")
print(f"  VADER took {vader_time:.4f} seconds")
print(f"  RoBERTa took {roberta_time:.4f} seconds")
#CHECKING WHY VADER ASSIGNED 
# print(sentiment_tsv['11'], vader_model.polarity_scores(sentiment_tsv['11']['sentence']))
# for i in sentiment_tsv['11']['sentence'].split():
    
#     if i in vader_model.lexicon:
#         print(vader_model.lexicon[i], i)
#     if i not in vader_model.lexicon:
#         print(i)
#Checking if 'but' is at fault here
# print(vader_model.polarity_scores("The author’s writing style is so unique poetic - not over the top."))
