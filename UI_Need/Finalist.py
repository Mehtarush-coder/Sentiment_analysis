# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import pandas as pd
# from nltk.corpus import stopwords


# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def remove_empty_strings(strings):
#     """Removes empty strings from a list of strings.

#     Args:
#         strings (list): The list of strings to be cleaned.

#     Returns:
#         list: The list of strings without empty strings.
#     """

#     return [string for string in strings if string.strip()]


# def predict_sentiment(text):
#     # Split text into sentences based on special characters and punctuation
#     sentences = re.split(r'[.!?"]', text)

#     cleaned_strings = remove_empty_strings(sentences)
#     print(cleaned_strings)

#     # Analyze each sentence and collect sentiments
#     sentiments = []
#     for sentence in cleaned_strings:
#         inputs = tokenizer(sentence, return_tensors="pt")
#         outputs = model(**inputs)
#         predicted_logits = outputs.logits
#         predicted_class_id = predicted_logits.argmax().item()
#         predicted_label = model.config.id2label[predicted_class_id]
#         sentiments.append(predicted_label)

#     # Determine overall sentiment based on majority
#     sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in sentiments}
#     print(sentiment_counts)
#     most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#     return most_frequent_sentiment

# # Example usage
# text = "Beautiful flowers, reasonable price"
# predicted_sentiment = predict_sentiment(text)
# print(predicted_sentiment)  # Output: negative

# # <<<<<<<<<<<<<>>>>>>>>>





# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define error handling for potential exceptions during model loading
# try:
#   model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   model = AutoModelForSequenceClassification.from_pretrained(model_name)
# except OSError as e:
#   print(f"Error loading model: {e}")
#   exit(1)  # Exit with an error code if model loading fails


# #  <<<<<<<<<<<<<<<<<<<<<<<<< Remove_Empty_Strings >>>>>>>>>>>>>>>>>>>>>>>



# def remove_empty_strings(strings):
#   """Removes empty strings from a list of strings.

#   Args:
#       strings (list): The list of strings to be cleaned.

#   Returns:
#       list: The list of strings without empty strings.
#   """
#   return [string for string in strings if string.strip()]


# #  <<<<<<<<<<<<<<<<<<<<<<<<< Predict_Sentiment >>>>>>>>>>>>>>>>>>>>>>>


# def predict_sentiment(text):
#   """Predicts the overall sentiment of a text by analyzing individual sentences.

#   Args:
#       text (str): The text to analyze.

#   Returns:
#       str: The predicted overall sentiment (positive, negative, or neutral).
#   """
#   # Split text into sentences based on special characters and punctuation

#   sentences = re.split(r'[.!?"]', text)

#   cleaned_strings = remove_empty_strings(sentences)

#   # Analyze each sentence and collect sentiments

#   sentiments = []
#   for sentence in cleaned_strings:
#     try:
#       inputs = tokenizer(sentence, return_tensors="pt")
#       outputs = model(**inputs)
#       predicted_logits = outputs.logits
#       predicted_class_id = predicted_logits.argmax().item()
#       predicted_label = model.config.id2label[predicted_class_id]
#       sentiments.append(predicted_label)
#     except Exception as e:  # Catch any exceptions during prediction
#       print(f"Error predicting sentiment for sentence: '{sentence}': {e}")

#   # Determine overall sentiment based on majority
#   if not sentiments:  # Handle cases with no valid sentences
#     return "neutral"  # Default to neutral sentiment if no predictions

#   sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in sentiments}
#   most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#   return most_frequent_sentiment

# # Load your DataFrame (replace 'your_file.csv' with your actual path)

# try:
#   df = pd.read_csv(r'C:\Users\Parth Bhavnani\Desktop\RushabhMehta\NLPmodel.csv')
# except FileNotFoundError as e:
#   print(f"Error loading CSV file: {e}")
#   exit(1)  # Exit with an error code if file not found

# # Sentiment Distribution

# sentiment_counts = df['score'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.countplot(x='score', data=df)
# plt.title('Sentiment Distribution')
# plt.show()

# # Word Frequency Analysis (EDA)
# # Combine all reviews into a single string
# all_reviews = " ".join(df['review'])

# # Split the text into words, ignoring stop words and punctuation
# stop_words = set(stopwords.words('english'))
# words = [word for word in all_reviews.split() if word not in stop_words and word.isalnum()]

# # Count word frequencies
# word_counts = Counter(words)

# # Get the top 20 most frequent words
# most_common_words = word_counts.most_common(20)

# # Visualize word frequency
# plt.figure(figsize=(10, 6))
# plt.bar([word for word, count in most_common_words], [count for word, count in most_common_words])
# plt.title('Most Frequent Words')
# plt.xlabel('Word')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)
# plt.show()






# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
from nltk.corpus import stopwords

from tqdm import tqdm 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

# Define model name and load pre-trained DistilBERT model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def remove_empty_strings(strings):
    """Removes empty strings from a list of strings.

    Args:
        strings (list): The list of strings to be cleaned.

    Returns:
        list: The list of strings without empty strings.
    """

    return [string for string in strings if string.strip()]


def predict_sentiment(text):
    # Split text into sentences based on special characters and punctuation
    sentences = re.split(r'[.!?"]', text)

    cleaned_strings = remove_empty_strings(sentences)
    # print(cleaned_strings)  # For debugging

    # Analyze each sentence and collect sentiments
    sentiments = []
    for sentence in cleaned_strings:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        predicted_logits = outputs.logits
        predicted_class_id = predicted_logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        sentiments.append(predicted_label)

    # Determine overall sentiment based on majority
    sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in sentiments}
    # most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    # return most_frequent_sentiment

    if sentiment_counts:  # Check if sentiment_counts is not empty
      most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    else:
      most_frequent_sentiment = "neutral"  # Default sentiment for empty dictionary

    return most_frequent_sentiment

# Load the CSV file
df = pd.read_csv(r'C:\Users\Parth Bhavnani\Desktop\RushabhMehta\NLPmodel.csv')

# Add a new column for predicted sentiment (optional)
if 'predicted_sentiment' not in df.columns:
    df['predicted_sentiment'] = None


total_reviews = len(df)
pbar = tqdm(total=total_reviews)

ground_truth = []
predictions = []


# Iterate through each row and analyze sentiment
for index, row in df.iterrows():
    review_text = row['review']
    predicted_sentiment = predict_sentiment(review_text)
    

    df.at[index, 'predicted_sentiment'] = predicted_sentiment

    if df.at[index,'review'] == 'undefined':
        df.at[index, 'predicted_sentiment'] = "NEUTRAL"
    
    ground_truth.append(row['predicted_sentiment'])  # Assuming 'sentiment' column holds actual labels
    predictions.append(predicted_sentiment)

     # Update progress bar and display percentage
    pbar.update(1)
    completed = (index + 1) / total_reviews * 100
    pbar.set_description(f"Progress: {completed:.2f}%")

pbar.close()  

# Save the updated DataFrame to a CSV file (specify the output path)
df.to_csv(r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv', index=False)

# Print the DataFrame with the added column (optional)
print(df.head(50))


# <<<<<<<<<<<<<<<<<<<<< TOP 5 SUPPLIER NAME BASE ON POSITIVE SENTIMENTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>


import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df' and has columns 'supplier_name' and 'predicted_sentiment'

# Filter non-undefined reviews
filtered_df = df[df['predicted_sentiment'] != "NEUTRAL"]  # Assuming "NEUTRAL" indicates undefined

# Filter positive sentiment reviews
positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == "POSITIVE"]

# Group by supplier name and count positive reviews
positive_reviews_by_supplier = positive_reviews.groupby('supplier_name').size()

# Sort by count in descending order (most positive to least)
top_5_suppliers = positive_reviews_by_supplier.sort_values(ascending=False).head(5)

# Check if any top 5 have zero reviews (indicating no positive reviews)
if 0 in top_5_suppliers.values:
    # Filter out suppliers with zero reviews (optional)
    top_5_suppliers = top_5_suppliers[top_5_suppliers != 0]

# Create a bar chart (assuming top_5_suppliers has valid entries)
plt.figure(figsize=(10, 6))
plt.bar(top_5_suppliers.index, top_5_suppliers.values)
plt.xlabel('Supplier Name')
plt.ylabel('Number of Positive Reviews')
plt.title('Top 5 Suppliers with Highest Positive Sentiment')
plt.xticks(rotation=45)
plt.show()




# <<<<<<<<<<<<<<<<<<< SENTIMENT WORD CLOUD >>>>>>>>>>>>>>>>>>>>>>>>

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create separate word clouds for positive, negative, and neutral sentiments
positive_words = " ".join(df[df['predicted_sentiment'] == 'POSITIVE']['review'])
negative_words = " ".join(df[df['predicted_sentiment'] == 'NEGATIVE']['review'])
neutral_words = " ".join(df[df['predicted_sentiment'] == 'NEUTRAL']['review'])

# Generate word clouds
wordcloud_positive = WordCloud(width=800, height=400).generate(positive_words)
wordcloud_negative = WordCloud(width=800, height=400).generate(negative_words)
wordcloud_neutral = WordCloud(width=800, height=400).generate(neutral_words)

# Display word clouds
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Sentiment')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Sentiment')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.title('Neutral Sentiment')
plt.axis('off')

plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame has columns 'sentiment' and 'review'
df['text_length'] = df['review'].apply(len)
df['predicted_sentiment'] = df['predicted_sentiment'].str.lower()
df['predicted_sentiment'] = df['predicted_sentiment'].replace("neutral", "NEUTRAL")

# Group by sentiment and calculate average text length
avg_text_length_by_sentiment = df.groupby('predicted_sentiment')['text_length'].mean()

# Visualize the correlation
sns.barplot(x=avg_text_length_by_sentiment.index, y=avg_text_length_by_sentiment.values)
plt.xlabel('Sentiment')
plt.ylabel('Average Text Length')
plt.title('Sentiment Correlation with Text Length')
plt.show()


import pandas as pd

# Assuming your DataFrame is named 'df' and the column containing supplier names is 'supplier_name'

# Count occurrences of each unique supplier name
supplier_counts = df['supplier_name'].value_counts()

# Get the top 5 suppliers with the highest counts
top_5_suppliers = supplier_counts.head(5)

# Print the results
print(top_5_suppliers)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_5_suppliers.index, top_5_suppliers.values)
plt.xlabel('Supplier Name')
plt.ylabel('Count')
plt.title('Top 5 Suppliers')
plt.xticks(rotation=45)
plt.show()



