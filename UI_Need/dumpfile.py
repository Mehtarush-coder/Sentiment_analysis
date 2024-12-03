# from django.shortcuts import render

# # Create your views here.
# from django.shortcuts import render
# from django.http import HttpResponseRedirect
# from .forms import FileUploadForm
# import pandas as pd
# # from . Finalist import 
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import pandas as pd
# from nltk.corpus import stopwords

# from tqdm import tqdm 
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

# def index(request):
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#     import re
#     import pandas as pd
#     from nltk.corpus import stopwords

#     from tqdm import tqdm 
#     from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
#     if request.method == 'POST':
#         form = FileUploadForm(request.POST, request.FILES)
#         positive_count = 0  # Initialize variables outside the 'if' block
#         negative_count = 0
#         neutral_count = 0
#         show_results = False
#         Total_review = 0
#         positive_count_percentage=0
#         negative_count_percentage=0
#         neutral_count_percentage=0


#         if form.is_valid():
#             excel_file = request.FILES['excel_file']

            
#             # Define model name and load pre-trained DistilBERT model
#             model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForSequenceClassification.from_pretrained(model_name)

#             def remove_empty_strings(strings):
#                 """Removes empty strings from a list of strings.

#                 Args:
#                     strings (list): The list of strings to be cleaned.

#                 Returns:
#                     list: The list of strings without empty strings.
#                 """

#                 return [string for string in strings if string.strip()]


#             def predict_sentiment(text):
#                 # Split text into sentences based on special characters and punctuation
#                 sentences = re.split(r'[.!?"]', text)

#                 cleaned_strings = remove_empty_strings(sentences)
#                 # print(cleaned_strings)  # For debugging

#                 # Analyze each sentence and collect sentiments
#                 sentiments = []
#                 for sentence in cleaned_strings:
#                     inputs = tokenizer(sentence, return_tensors="pt")
#                     outputs = model(**inputs)
#                     predicted_logits = outputs.logits
#                     predicted_class_id = predicted_logits.argmax().item()
#                     predicted_label = model.config.id2label[predicted_class_id]
#                     sentiments.append(predicted_label)


#                 # Determine overall sentiment based on majority
#                 sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in sentiments}
#                 # most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#                 # return most_frequent_sentiment

#                 if sentiment_counts:  # Check if sentiment_counts is not empty
#                     most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#                 else:
#                     most_frequent_sentiment = "neutral"  # Default sentiment for empty dictionary

#                 return most_frequent_sentiment

#             # Load the CSV file
#             df = pd.read_csv(excel_file)
#             Total_review = len(df)
#             print("Total_review",Total_review)

#             if 'predicted_sentiment' in df.columns:
                    
#                     print("okokokokokokokokok")
#                     positive_count = df[df['predicted_sentiment'] == 'POSITIVE'].shape[0]
#                     negative_count = df[df['predicted_sentiment'] == 'NEGATIVE'].shape[0]
#                     neutral_count = df[df['predicted_sentiment'] == 'NEUTRAL'].shape[0]
#                     positive_count_percentage = ((positive_count/Total_review)*100)
#                     print("positive_count_percentage",positive_count_percentage)
#                     negative_count_percentage = (negative_count/Total_review)*100
#                     neutral_count_percentage = (neutral_count/Total_review)*100
#                     show_results = True
#             else:
#                     # Handle the case where the 'sentimental_analysis' column doesn't exist
#                 pass

#             # Add a new column for predicted sentiment (optional)
#             if 'predicted_sentiment' not in df.columns:
#                 df['predicted_sentiment'] = None


#             total_reviews = len(df)
#             pbar = tqdm(total=total_reviews)

#             ground_truth = []
#             predictions = []

#             # Iterate through each row and analyze sentiment
#             for index, row in df.head(50).iterrows():
#                 review_text = row['review']
#                 predicted_sentiment = predict_sentiment(review_text)
                

#                 df.at[index, 'predicted_sentiment'] = predicted_sentiment

#                 if df.at[index,'review'] == 'undefined':
#                     df.at[index, 'predicted_sentiment'] = "NEUTRAL"
                
#                 ground_truth.append(row['predicted_sentiment'])  # Assuming 'sentiment' column holds actual labels
#                 predictions.append(predicted_sentiment)

#                 # Update progress bar and display percentage
#                 pbar.update(1)
#                 completed = (index + 1) / total_reviews * 100
#                 pbar.set_description(f"Progress: {completed:.2f}%")

#             pbar.close()  

#             # Save the updated DataFrame to a CSV file (specify the output path)
#             df.to_csv(r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv', index=False)

#             # total_reviews = len(df)
#             # pbar = tqdm(total=total_reviews, desc='Analyzing Reviews')

#             # Print the DataFrame with the added column (optional)
#             print(df.head(50))


#             # <<<<<<<<<<<<<<<<<<<<< TOP 5 SUPPLIER NAME BASE ON POSITIVE SENTIMENTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#             import matplotlib.pyplot as plt

#             # Assuming your DataFrame is named 'df' and has columns 'supplier_name' and 'predicted_sentiment'

#             # Filter non-undefined reviews
#             filtered_df = df[df['predicted_sentiment'] != "NEUTRAL"]  # Assuming "NEUTRAL" indicates undefined

#             # Filter positive sentiment reviews
#             positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == "POSITIVE"]

#             # Group by supplier name and count positive reviews
#             positive_reviews_by_supplier = positive_reviews.groupby('supplier_name').size()

#             # Sort by count in descending order (most positive to least)
#             top_5_suppliers = positive_reviews_by_supplier.sort_values(ascending=False).head(5)

            
#             # Check if any top 5 have zero reviews (indicating no positive reviews)
#             if 0 in top_5_suppliers.values:
#                 # Filter out suppliers with zero reviews (optional)
#                 top_5_suppliers = top_5_suppliers[top_5_suppliers != 0]

#             import io
#             import base64
#             from django.core.serializers.json import DjangoJSONEncoder
#             plt.switch_backend('Agg')            # Create a bar chart (assuming top_5_suppliers has valid entries)
#             plt.figure(figsize=(10, 6))
#             plt.bar(top_5_suppliers.index, top_5_suppliers.values)
#             plt.xlabel('Supplier Name')
#             plt.ylabel('Number of Positive Reviews')
#             plt.title('Top 5 Suppliers with Highest Positive Sentiment')
#             plt.xticks(rotation=45)

#             # Create a buffer to capture the plot image data
#             buffer = io.BytesIO()

#             # Save the plot to the buffer in PNG format (adjust format as needed)
#             plt.savefig(buffer, format='png')

#             # Close the plot figure to release resources
#             plt.close()

#             # Encode the image data in base64 format for transmission in JSON
#             image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

#             print("image_data",image_data)


#             # Create a dictionary to store plot information
#             plot_data = {
#                 'image_data': image_data,  # Base64 encoded image data
#                 'labels': list(top_5_suppliers.index),  # Supplier names
#                 'values': list(map(int, top_5_suppliers.values)),  # Count of positive reviews
#                 'xlabel': 'Supplier Name',  # X-axis label
#                 'ylabel': 'Number of Positive Reviews',  # Y-axis label
#                 'title': 'Top 5 Suppliers with Highest Positive Sentiment',  # Chart title
#             }

#             # print("plot_data",plot_data)

#             # Serialize the plot data dictionary to JSON using DjangoJSONEncoder
#             plot_json = DjangoJSONEncoder().encode(plot_data)

#             print("plot_data",plot_json)

#             return render(request, 'index.html', {
#                 'form': form, 
#                 'positive_count': positive_count, 'negative_count': negative_count, 'neutral_count': neutral_count,'show_results': show_results ,'top_5_suppliers':top_5_suppliers,
#                  "positive_count_percentage":positive_count_percentage,"negative_count_percentage":negative_count_percentage , "neutral_count_percentage":neutral_count_percentage,
#                   'plot_json': plot_json # Optional: base64 encoded image
#                 # Add similar context for other visualizations
#             })

            # <<<<<<<<<<<<<<<<<<< SENTIMENT WORD CLOUD >>>>>>>>>>>>>>>>>>>>>>>>

    #         from wordcloud import WordCloud
    #         import matplotlib.pyplot as plt

    #         # Create separate word clouds for positive, negative, and neutral sentiments
    #         positive_words = " ".join(df[df['predicted_sentiment'] == 'POSITIVE']['review'])
    #         negative_words = " ".join(df[df['predicted_sentiment'] == 'NEGATIVE']['review'])
    #         neutral_words = " ".join(df[df['predicted_sentiment'] == 'NEUTRAL']['review'])

    #         # Generate word clouds
    #         wordcloud_positive = WordCloud(width=800, height=400).generate(positive_words)
    #         wordcloud_negative = WordCloud(width=800, height=400).generate(negative_words)
    #         wordcloud_neutral = WordCloud(width=800, height=400).generate(neutral_words)

    #         # Display word clouds
    #         plt.figure(figsize=(15, 10))
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(wordcloud_positive, interpolation='bilinear')
    #         plt.title('Positive Sentiment')
    #         plt.axis('off')

    #         plt.subplot(1, 3, 2)
    #         plt.imshow(wordcloud_negative, interpolation='bilinear')
    #         plt.title('Negative Sentiment')
    #         plt.axis('off')

    #         plt.subplot(1, 3, 3)
    #         plt.imshow(wordcloud_neutral, interpolation='bilinear')
    #         plt.title('Neutral Sentiment')
    #         plt.axis('off')

    #         plt.show()

    #         import seaborn as sns
    #         import matplotlib.pyplot as plt

    #         # Assuming your DataFrame has columns 'sentiment' and 'review'
    #         df['text_length'] = df['review'].apply(len)
    #         df['predicted_sentiment'] = df['predicted_sentiment'].str.lower()
    #         df['predicted_sentiment'] = df['predicted_sentiment'].replace("neutral", "NEUTRAL")

    #         # Group by sentiment and calculate average text length
    #         avg_text_length_by_sentiment = df.groupby('predicted_sentiment')['text_length'].mean()

    #         # Visualize the correlation
    #         sns.barplot(x=avg_text_length_by_sentiment.index, y=avg_text_length_by_sentiment.values)
    #         plt.xlabel('Sentiment')
    #         plt.ylabel('Average Text Length')
    #         plt.title('Sentiment Correlation with Text Length')
    #         plt.show()


    #         import pandas as pd

    #         # Assuming your DataFrame is named 'df' and the column containing supplier names is 'supplier_name'

    #         # Count occurrences of each unique supplier name
    #         supplier_counts = df['supplier_name'].value_counts()

    #         # Get the top 5 suppliers with the highest counts
    #         top_5_suppliers = supplier_counts.head(5)

    #         # Print the results
    #         print(top_5_suppliers)

    #         import matplotlib.pyplot as plt

    #         plt.figure(figsize=(10, 6))
    #         plt.bar(top_5_suppliers.index, top_5_suppliers.values)
    #         plt.xlabel('Supplier Name')
    #         plt.ylabel('Count')
    #         plt.title('Top 5 Suppliers')
    #         plt.xticks(rotation=45)
    #         plt.show()

    #         # # Process the uploaded Excel file here
    #         # df = pd.read_excel(excel_file)
    #         # # Run your Python script using the DataFrame
    #         # # ...

    #         return HttpResponseRedirect('/')  # Redirect to the home page or another view
    # else:
    #     form = FileUploadForm()
    # return render(request, 'index.html', {'form': form})

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import FileUploadForm
import pandas as pd
# from . Finalist import 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
from nltk.corpus import stopwords

from tqdm import tqdm 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
import matplotlib.pyplot as plt
import io

import base64


def create_plot():
    
    global excel_file
    excel_file = (r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv')

    import matplotlib.pyplot as plt

    # Assuming your DataFrame is named 'df' and has columns 'supplier_name' and 'predicted_sentiment'

    # Filter non-undefined reviews
    df = pd.read_csv(excel_file)
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

    import io
    import base64
    from django.core.serializers.json import DjangoJSONEncoder
    plt.switch_backend('Agg')            # Create a bar chart (assuming top_5_suppliers has valid entries)
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_suppliers.index, top_5_suppliers.values)
    plt.xlabel('Supplier Name')
    plt.ylabel('Number of Positive Reviews')
    plt.title('Top 5 Suppliers with Highest Positive Sentiment')
    plt.xticks(rotation=45)

    # Create a buffer to capture the plot image data
    buffer = io.BytesIO()

    # Save the plot to the buffer in PNG format (adjust format as needed)
    plt.savefig(buffer, format='png')

    # Close the plot figure to release resources
    plt.close()

    # Encode the image data in base64 format for transmission in JSON
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_data

def index(request):
    plot_data = create_plot()
    context = {'plot_data': plot_data}
    return render(request, 'index.html', context)

