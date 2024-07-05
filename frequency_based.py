import pandas as pd
from collections import Counter

#%%
def calculate_metric(df, labels_words, doc_col, label_col):
    """
    Calculate the specified metric for each document and its labels.

    Parameters:
    df (pd.DataFrame): DataFrame containing documents and labels.
    doc_col (str): Column name of the documents.
    label_col (str): Column name of the labels.

    Returns:
    pd.Series: A series containing the metric for each document.
    """

    # Flatten list of documents and labels to count global term frequencies
    all_words = [word for doc in df[doc_col] for word in doc.split()]

    word_freq_in_all_docs = Counter(all_words)
    word_freq_in_all_labels = Counter(labels_words)

    metrics = []

    # TODO change iterrow - it is very slow
    for _, row in df.iterrows():
        doc_words = row[doc_col].split()
        label_words = row[label_col].split()

        doc_word_counts = Counter(doc_words)
        label_word_counts = Counter(label_words)

        term_freq_in_body = sum((doc_word_counts[word] / word_freq_in_all_docs[word]) for word in doc_words if
                                word in word_freq_in_all_docs)
        term_freq_in_labels = sum(
            (label_word_counts[word] / len(label_words)) for word in label_words if word in word_freq_in_all_labels)

        metric = term_freq_in_body * term_freq_in_labels
        metrics.append(metric)

    return pd.Series(metrics)

def read_freq_words(freq_words_file):
    with open(freq_words_file, 'r') as f:
        words = f.read().splitlines()
    return words

#%%

dataset_file = "/data/data/train_train.csv"
labels_words = read_freq_words("frequent.txt")
df = pd.read_csv(dataset_file)

#%%

# #if __name__=="main":
#     labels_words =

df['freq_metric'] = calculate_metric(df, labels_words, 'body', 'label')
# print(df)
