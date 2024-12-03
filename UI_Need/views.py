from django.shortcuts import render
import matplotlib
matplotlib.use('Agg') 
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
import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
import base64
import csv
global top_5_suppliers
top_5_suppliers = None
import matplotlib.pyplot as plt
import io   
import base64
from base64 import b64encode
from django.core.serializers.json import DjangoJSONEncoder
from wordcloud import WordCloud
import base64
from io import BytesIO
from collections import Counter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import StreamingHttpResponse

product_quality_keywords = [
    "vibrant", "beautiful", "fresh","great condition", "lush", "blooming", "lively", "colorful", "eye-catching", "well-arranged", "pristine",
    "radiant", "vivid", "gorgeous", "stunning", "immaculate", "crisp", "dewy", "healthy-looking", "blooming beautifully", "full of life",
    "sweet", "pleasant", "fragrant", "aromatic", "delicate", "strong", "natural", "refreshing", "enduring", "soothing",
    "heavenly", "intoxicating", "enticing", "uplifting", "calming","lasted longer","Great flowers","loves","love","loved",
    "long-lasting", "durable", "fresh-looking", "healthy", "sturdy", "robust", "enduring", "resilient", "wither-resistant", "long-blooming",
    "lasting", "resilient", "durable", "long-lasting", "enduring","not been satisfied","pretty","loved the flowers ",
    "well-packaged", "secure", "attractive", "presentable", "neat", "tidy", "professional", "thoughtful", "impeccable", "exquisite",
    "elegant", "sophisticated", "wrapped", "beautifully presented"," Bouqs is just the best","absolutely lovely","blooms",
    "high-quality","luxurious", "sumptuous", "opulent", "magnificent", "splendid", "glorious", "marvelous", "wonderful","Great product", "fantastic", "amazing", "perfect"
]

product_quality_negative_keywords = [
    "rough shape","wilted", "wilt","die","dying","dead","worst quality","bruising","quality is lacking","died","brown", "dried", "faded", "unfresh", "old", "stale", "discolored", "unhealthy",
    "musty", "rotten", "unnatural", "short-lived", "perishable", "fragile", "weak", "flimsy", "low-quality", "inferior", "subpar", "unacceptable", "unsatisfactory",
    "poorly-packaged", "damaged", "broken", "crushed", "unorganized", "messy", "unprofessional", "sloppy", "careless", "negligent",
    "disappointing", "unsatisfactory", "unhappy", "upset", "angry", "frustrated", "dismayed", "disheartened", "let down",
    "substandard", "mediocre", "second-rate", "poor quality","Poor Quality", "bad","small","wimpy","lame","Horrible flowers",
    "unappealing", "ugly", "unattractive","tiny","not very nice","looked worse","little bit flustered"," large thorns","not very big",
    "faded", "dying", "dead", "brown", "unhealthy", "sickly", "diseased", "decayed","was not pleased with the quality","Uneven quality",
    "unnatural", "artificial", "synthetic", "fake", "imitation", "counterfeit", "forged", "phoney", "bogus", "sham","did not look","not in bloom","didn't bloom"
    "low-quality","flawed", "defective", "imperfect", "faulty", "broken", "damaged", "damaged", "ruined", "spoiled", "mutilated","ordinary", "common", 
    "outdated", "old-fashioned","wilt","wiltering","Wilting Petal","lack of flower","Quality","dying","ripoff","old","Not fresh","drooping down","beaten",
    "worn","no card","dark","curled","proflower bouquet","liliies","frozen marks","dark blotches","Squished","flower food","squished","ass products",
    "ugliest","hottendous","limp","small buds","droopy","bruished","ratty","crumpled","weren't as fresh","will they last"," not pretty","didn't last","splotches"
]

customer_service_keywords = [
    "responsive", "courteous", "informative", "communicative", "accessible","like the service",
    "efficient", "effective", "solution-oriented", "problem-solving", "accommodating", "understanding", "compassionate", "empathetic", "patient", "supportive",
    "personalized", "customized", "tailor-made", "attentive to detail","caring","customer service",
    "excellent", "superior", "impressed", "satisfied", "delighted", "happy", "flawless", "seamless","did not provide","service",
    "professional", "reliable", "service was prompt","friendly", "pleasant", "helpful", "cooperative", "enthusiastic", "passionate","Very prompt service"
]

customer_service_negative_keywords = [
"unresponsive", "rude", "unhelpful", "discourteous","customer service", "misleading", "uncommunicative", "inaccessible", "unavailable", "delayed", "slow",
"inefficient", "ineffective", "unhelpful", "problem-ignoring", "unaccommodating", "unsympathetic", "uncaring", "unempathetic", "impatient", "unsupportive",
"unpersonalized", "generic", "template-based", "inattentive", "thoughtless", "indifferent", "disrespectful", "ununderstanding", "inconsiderate",
"disappointing", "unsatisfactory", "unimpressed", "dissatisfied", "upset", "angry", "frustrated", "dismayed", "disheartened", "let down",
"unprofessional", "ignorant", "untrustworthy", "unreliable", "unfriendly", "unpleasant", "uncooperative", "unenthusiastic", "unpassionate",
"difficult", "challenging", "problematic", "frustrating", "stressful", "unsatisfactory", "unpleasant", "negative", "bad", "awful",
"arrogant", "condescending", "patronizing", "dismissive", "sarcastic", "rude", "insulting", "offensive", "abusive", "harmful",
"uncooperative", "unhelpful", "obstructive", "unwilling", "reluctant", "resistant", "defiant", "recalcitrant", "intractable", "stubborn",
"incompetent", "inept", "incapable", "unqualified", "unskilled", "unfit", "unsuited", "inappropriate", "irrelevant", "out of place",
"unorganized", "disorganized", "chaotic", "messy", "sloppy", "careless", "negligent", "irresponsible", "unprofessional", "unacceptable",
"slow", "sluggish", "lethargic", "lackadaisical", "indifferent", "uninterested", "apathetic", "uncaring", "unconcerned", "unsympathetic",
"unresponsive", "uncommunicative", "unwilling", "reluctant", "resistant", "defiant", "recalcitrant", "intractable", "stubborn",
"unhelpful", "obstructive", "uncooperative", "unsupportive","customer service was not","Hard to search","no communication","service","can't answer"
]

purchase_experience_positive_keywords = [
"seamless", "easy", "hassle-free", "convenient", "intuitive", "user-friendly", "straightforward", "efficient", "quick", "fast",
"clear", "informative", "transparent", "honest", "trustworthy", "reliable", "secure", "safe", "protected", "private",
"diverse", "varied", "wide-ranging", "extensive", "abundant", "many", "lots", "plenty", "choices", "options",
"satisfied","thrilled","grateful", "thankful","efficient customer service","was very happy","reccomended","already referred",
"stress-free", "effortless","fantastic experience","simple", "uncomplicated", "smooth", "seamless","satisfactory", "good", "great", "excellent","Easy to order"
]

purchase_experience_negative_keywords = [
    "website design is a tragedy",
    "look nothing like",
    "embarrassed",
    "Forgot",
    "not assembled",
    "couldn't find",
    "Email blasts",
    "delivery communication",
    "presentation",
    "Service",
    "online",
    "Photograph",
    "stopping",
    "subscription",
    "adverstised",
    "arrangement",
    "promotions",
    "difficult",
    "frustrating",
    "complicated",
    "cumbersome",
    "tedious",
    "confusing",
    "unclear",
    "unintuitive",
    "incorrect order",
    "reccomended",
    "not positive",
    "limited",
    "restricted",
    "narrow",
    "few",
    "insufficient",
    "inadequate",
    "lacking",
    "deficient",
    "sparse",
    "scarce",
    "Flowers didn't come",
    "Not as big",
    "embarrassing",
    "unpersonalized",
    "generic",
    "template-based",
    "inattentive",
    "thoughtless",
    "indifferent",
    "disrespectful",
    "ununderstanding",
    "inconsiderate",
    "wrong card",
    "wrap wasn't the same",
    "poorly-packaged",
    "negligent",
    "affordable prices",
    "disappointment",
    "embarrassment",
    "dont completely have faith",
    "did not sent",
    "choose on the website did not look",
    "late",
    "delayed",
    "slow",
    "inefficient",
    "unreliable",
    "unpredictable",
    "inaccurate",
    "imprecise",
    "received were not like",
    "don't look like the picture",
    "canceled my order",
    "unsatisfied",
    "did not feel",
    "unhappy",
    "displeased",
    "upset",
    "angry",
    "disappointed",
    "mutilated",
    "awful",
    "terrible",
    "barely even",
    "Didn't look like the picture",
    "What you order is not what you get",
    "hardly romantic",
    "Past experiences",
    "Not really on expectation",
    "False advertising",
    "nothing like the photo",
    "received were not",
    "coupon away",
    "not sure how the quality",
    "quantity wasn't the same",
    "haven't found a location",
    "flowers weren't quite",
    "something else",
    "wrong flower",
    "cancelled",
    "Didn't look",
    "requested to only",
    "not as pictured",
]

pricing_positive_keywords = [
"affordable", "reasonable", "cost-effective", "good value", "worth it", "great deal", "bargain", "discount", "sale","transparent",
"value for money", "bang for your buck", "worth every penny", "money well spent", "excellent value", "high quality for the price",
"fair", "just", "equitable", "reasonable", "budget-friendly", "inexpensive", "cheap", "discounted", "sale",
"competitive", "comparable", "similar", "equivalent", "matched", "equal", "on par", "in line with", "comparable to", "similar to",
"good value", "worth it", "a bargain", "a steal", "a great deal", "a good deal", "a good price", "a fair price",
"well-priced", "reasonably priced", "inexpensively priced", "cheaply priced", "discounted priced", "sale priced","worth the price"
"cost-effective","economical","Great price"," reasonably priced","do not have to pay extra","worth the money","not pricy","no service charges"," saves money",
"well-spent", "worthwhile", "valuable","No hidden costs","good prices"]

pricing_negative_keywords = [
"expensive", "overprice", "costly","pay extra", "unaffordable", "unreasonable", "not worth it", "scam", "fraudulent",
"hidden fees", "extra charges", "surcharges", "unexpected costs", "additional costs", "surprise costs", "unforeseen costs",
"poor value", "bad deal", "not worth the money",
"unfair","hidden fees", "extra charges", "surcharges", "unexpected costs", "additional costs", "surprise costs", "unforeseen costs",
"price gouging", "price hiking", "inflationary", "cost-increasing", "price-raising",
"unaffordable","costly", "expensive","high-priced", "premium-priced","High price"
"absurd", "ridiculous", "unjustified", "unfair", "unjust", "unequal", "inappropriate", "unreasonable", "exorbitant", "outrageous",
"hidden fees", "extra charges", "surcharges", "unexpected costs", "additional costs", "surprise costs", "unforeseen costs",
"price gouging", "price hiking", "inflationary", "cost-increasing", "price-raising","pricey"
"inflated", "exaggerated", "overstated", "unrealistic", "unbelievable",
"poor value", "bad deal", "not worth the money","waste of money","worth the money"," underwhelming for the price",
"disappointing","worth the price","No hidden prices"
]

delivery_positive_keywords = [
"fast delivery", "quick", "timely", "efficient","same day shipping","prompt", "on-time", "punctual", "scheduled", "accurate", "precise",
"safe", "secure", "protected", "preserved", "delighted", "thrilled", "amazed", "impressed", "content", "grateful", "thankful",
"dependable", "trustworthy", "assured", "certain", "guaranteed","delivery went very well","delivery","deliver"
"safe", "secure", "protected", "preserved", "intact", "unharmed", "undamaged","delivery is reliable",
"effortless", "uncomplicated", "smooth","fast delivery","Fast delivery","on time","delivery is tracked on","Free shipping",
"on-time", "scheduled", "as promised", "when expected", "right on time", "dead on time", "exactly on time",
"quick", "fast", "rapid", "speedy", "swift", "expeditious", "timely","reliable delivery","delivered"
"reliable", "predictable", "consistent", "dependable", "trustworthy", "assured", "certain", "guaranteed","delivery was on time",
"safe", "secure", "protected", "preserved", "intact", "unharmed", "undamaged", "in good condition", "well-maintained","delivered on time",
"tracked", "monitored", "informed", "notified","prompt delivery", "kept in the loop","arrived timely","arrived on time"," get here on time",
"On time delivery", "content", "gratified", "delighted", "thrilled","arrived as promised","delivery was fast","delivery on time","Always on time","arrived","arrive"
]

delivery_negative_keywords = [
"slow", "delayed", "late","untimely","haven't received","never delivered","unable to deliver","late", "unpunctual", "unscheduled","imprecise", "unorganized", "messy", "unprofessional", "sloppy", "careless",
"lost", "missing", "stolen", "wrong address", "misdelivered","delivered","delivery","deliver","never received","received","not receive flowers",
"unmonitored", "unupdated", "uninformed", "unnotified","not arrived","order took too long",
"delayed", "late", "tardy", "behind schedule", "untimely", "unpunctual", "unseasonable", "out of place", "out of time", "untimely",
"missed", "failed to arrive", "did not arrive", "were not delivered"," not delivered", "were not there", "were not present",
"lost", "missing", "misplaced", "gone", "vanished", "disappeared", "abducted", "kidnapped", "stolen", "robbed", "scratched","Delays orders",  
"didn't arrive","Shipping delay","wrong place","waited for delivery","did not receive the flowers","never received","never came"
"Delivery lost","wrong place","not received","Late delivery","delivered a day late","haven't been delivered","Didn't arrive", "problems with delivery","haven't arrived"
]


FedEx = ["blamed","problems","inconvenience","shipping issues","disarray","Fed Ex","blame","screwed up","hiccup","harsh",
         "sucked","showed up late","signature"]


# def count_keywords(review, keyword_dict):
#     word_counts = {}
#     for department, keywords in keyword_dict.items():
#         for keyword in keywords["positive"]:
#             count = review.lower().count(keyword)
#             if count > 0:
#                 word_counts[keyword] = {"count": count, "department": department, "sentiment": "positive"}
#         for keyword in keywords["negative"]:
#             count = review.lower().count(keyword)
#             if count > 0:
#                 word_counts[keyword] = {"count": count, "department": department, "sentiment": "negative"}
#     return word_counts




def count_keywords(review, keyword_dict):
    word_counts = {}
    
    if not isinstance(review, str):
        return word_counts  # Return an empty dict if not a string

    for department, keywords in keyword_dict.items():
        for keyword in keywords["positive"]:
            count = review.lower().count(keyword)
            if count > 0:
                word_counts[keyword] = {"count": count, "department": department, "sentiment": "positive"}
        for keyword in keywords["negative"]:
            count = review.lower().count(keyword)
            if count > 0:
                word_counts[keyword] = {"count": count, "department": department, "sentiment": "negative"}
    return word_counts


def departments(df,years):
    keyword_dict = {
    "product_quality": {
        "positive": product_quality_keywords,
        "negative": product_quality_negative_keywords
    },
    "customer_service": {
        "positive": customer_service_keywords,
        "negative": customer_service_negative_keywords
    },
    "purchase_experience": {
        "positive": purchase_experience_positive_keywords,
        "negative": purchase_experience_negative_keywords
    },
    "pricing": {
        "positive": pricing_positive_keywords,
        "negative": pricing_negative_keywords
    },
    "delivery": {
        "positive": delivery_positive_keywords,
        "negative": delivery_negative_keywords
    },
    "FedEx":{
        "positive" : FedEx,
        "negative" : FedEx 
    }
}
    
    results=[]

    # sentiment_analysis_df = pd.read_csv(excel_file, encoding='utf-8') 

    # with open(excel_file, "r") as csvfile:
    #     reader = csv.DictReader(csvfile)

    # for i, row in enumerate(reader):

    df = convert_timestamp(df.copy())
    
    print("Selected years:", years)

   # Filter data for the selected years
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]


    for index, row in filtered_df.iterrows():

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<# need to convert it into length  of file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if index >= len(filtered_df):
            break
        review = row["review"]  # Assuming the review column is named "review"
        review_counts = count_keywords(review, keyword_dict)


        # Create a new row for the output CSV
        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]  # Store the count
            output_row[f"{keyword}_department"] = data["department"]  # Store the department

        results.append(output_row)
    df_data = []
    serial_numbers = []
    for result in results:
        review = result["review"]
        department_counts = {
            "product_quality": 0,
            "customer_service": 0,
            "purchase_experience": 0,
            "pricing": 0,
            "delivery": 0,
            "FedEx":0
        }

        for keyword, data in result.items():
            if keyword.endswith("_department"):
                department = data
                department_counts[department] = 1

        # Add the review and department counts to the DataFrame
        df_data.append({"review": review, **department_counts})
        
    df_new = pd.DataFrame(df_data)

    # Count output dataframe

    df_new['S_No'] = range(0, len(df_new))
    

    # Sentiment dataframe

    filtered_df['S_No'] = range(0, len(df))

    merged_df = pd.merge(df_new, filtered_df, on="S_No")

    # Initialize counters
    count_positive_product_quality = 0
    count_positive_customer_service = 0
    count_positive_purchase_experience = 0
    count_positive_pricing = 0
    count_positive_delivery = 0
    count_positive_FedEx= 0

    count_negative_product_quality = 0
    count_negative_customer_service = 0
    count_negative_purchase_experience = 0
    count_negative_pricing = 0
    count_negative_delivery = 0
    count_negative_FedEx= 0

    for index, row in merged_df.iterrows():

        if row["predicted_sentiment"] == "POSITIVE":
            if row["product_quality"] == 1:
                count_positive_product_quality += 1
            if row["customer_service"] == 1:
                count_positive_customer_service += 1
            if row["purchase_experience"] == 1:
                count_positive_purchase_experience += 1
            if row["pricing"] == 1:
                count_positive_pricing += 1
            if row["delivery"] == 1:
                count_positive_delivery += 1
            if row["FedEx"] == 1 :
                count_positive_FedEx +=1

        elif row["predicted_sentiment"] == "NEGATIVE":
            if row["product_quality"] == 1:
                count_negative_product_quality += 1
            if row["customer_service"] == 1:
                count_negative_customer_service += 1
            if row["purchase_experience"] == 1:
                count_negative_purchase_experience += 1
            if row["pricing"] == 1:
                count_negative_pricing += 1
            if row["delivery"] == 1:
                count_negative_delivery += 1
            if row["FedEx"] == 1 :
                count_negative_FedEx +=1



    return {
        "positive_product_quality": count_positive_product_quality,
        "positive_customer_service": count_positive_customer_service,
        "positive_purchase_experience": count_positive_purchase_experience,
        "positive_pricing": count_positive_pricing,
        "positive_delivery": count_positive_delivery,
        "negative_product_quality": count_negative_product_quality,
        "negative_customer_service": count_negative_customer_service,
        "negative_purchase_experience": count_negative_purchase_experience,
        "negative_pricing": count_negative_pricing,
        "negative_delivery": count_negative_delivery,
        "positive_FedEx":count_positive_FedEx,
        "negative_FedEx":count_negative_FedEx
    }, merged_df


import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
import base64

import matplotlib.pyplot as plt

def ProductQualityPositiveKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "product_quality": {
            "positive": product_quality_keywords,
            "negative": product_quality_negative_keywords
        }
    }

    results = []
    keyword_counts = {}  # Dictionary to keep track of keyword counts

    for index, row in df.iterrows():
        review = row["review"]  # Assuming the review column is named "review"
        review_counts = count_keywords(review, keyword_dict)

        # Create a new row for the output CSV
        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]  # Store the count
            output_row[f"{keyword}_department"] = data["department"]  # Store the department

            # Update the keyword counts for positive sentiments only
            if keyword in keyword_dict["product_quality"]["positive"]:
                if keyword in keyword_counts:
                    keyword_counts[keyword] += data["count"]
                else:
                    keyword_counts[keyword] = data["count"]

        results.append(output_row)

    # Get the top 5 keywords based on positive sentiment
    top_positive_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Convert top keywords into a more readable format if needed
    top_positive_keywords_dict = {keyword: count for keyword, count in top_positive_keywords}

    # Print the top keywords
    print("Top 5 Positive Keywords:", top_positive_keywords_dict)

    # Plotting the bar chart
    keywords = list(top_positive_keywords_dict.keys())
    counts = list(top_positive_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords, counts, color='skyblue')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Positive Keywords')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data




def ProductQualityNegativeKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "product_quality": {
            "positive": product_quality_keywords,
            "negative": product_quality_negative_keywords
        }
    }

    results = []
    negative_keyword_counts = {}  # Dictionary to keep track of negative keyword counts

    for index, row in df.iterrows():
        review = row["review"]  # Assuming the review column is named "review"
        review_counts = count_keywords(review, keyword_dict)

        # Create a new row for the output CSV
        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]  # Store the count
            output_row[f"{keyword}_department"] = data["department"]  # Store the department

            # Update the keyword counts for negative sentiments only
            if keyword in keyword_dict["product_quality"]["negative"]:
                if keyword in negative_keyword_counts:
                    negative_keyword_counts[keyword] += data["count"]
                else:
                    negative_keyword_counts[keyword] = data["count"]

        results.append(output_row)

    # Get the top 5 negative keywords
    top_negative_keywords = sorted(negative_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative_keywords_dict = {keyword: count for keyword, count in top_negative_keywords}

    # Print the top negative keywords
    print("Top 5 Negative Keywords:", top_negative_keywords_dict)

    # Plotting the horizontal bar chart for negative keywords
    keywords_negative = list(top_negative_keywords_dict.keys())
    counts_negative = list(top_negative_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords_negative, counts_negative, color='salmon')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Negative Keywords')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data  # Return the negative keyword chart


def CustomerServicePositiveKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "customer_service": {
            "positive": customer_service_keywords,  # Define your positive keywords for customer service
            "negative": customer_service_negative_keywords  # Define your negative keywords for customer service
        }
    }

    results = []
    keyword_counts = {}  # Dictionary to keep track of keyword counts

    for index, row in df.iterrows():
        review = row["review"]  # Assuming the review column is named "review"
        review_counts = count_keywords(review, keyword_dict)

        # Create a new row for the output CSV
        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]  # Store the count
            output_row[f"{keyword}_department"] = data["department"]  # Store the department

            # Update the keyword counts for positive sentiments only
            if keyword in keyword_dict["customer_service"]["positive"]:
                if keyword in keyword_counts:
                    keyword_counts[keyword] += data["count"]
                else:
                    keyword_counts[keyword] = data["count"]

        results.append(output_row)

    # Get the top 5 keywords based on positive sentiment
    top_positive_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Convert top keywords into a more readable format if needed
    top_positive_keywords_dict = {keyword: count for keyword, count in top_positive_keywords}

    # Print the top keywords
    print("Top 5 Positive Keywords for Customer Service:", top_positive_keywords_dict)

    # Plotting the bar chart
    keywords = list(top_positive_keywords_dict.keys())
    counts = list(top_positive_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords, counts, color='skyblue')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Positive Keywords for Customer Service')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data  # Return the generated chart


def CustomerServiceNegativeKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "customer_service": {
            "positive": customer_service_keywords,
            "negative": customer_service_negative_keywords
        }
    }

    results = []
    negative_keyword_counts = {}  # Dictionary to keep track of negative keyword counts

    for index, row in df.iterrows():
        review = row["review"]  # Assuming the review column is named "review"
        review_counts = count_keywords(review, keyword_dict)

        # Create a new row for the output CSV
        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]  # Store the count
            output_row[f"{keyword}_department"] = data["department"]  # Store the department

            # Update the keyword counts for negative sentiments only
            if keyword in keyword_dict["customer_service"]["negative"]:
                if keyword in negative_keyword_counts:
                    negative_keyword_counts[keyword] += data["count"]
                else:
                    negative_keyword_counts[keyword] = data["count"]

        results.append(output_row)

    # Get the top 5 negative keywords
    top_negative_keywords = sorted(negative_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative_keywords_dict = {keyword: count for keyword, count in top_negative_keywords}

    # Print the top negative keywords
    print("Top 5 Negative Keywords for Customer Service:", top_negative_keywords_dict)

    # Plotting the horizontal bar chart for negative keywords
    keywords_negative = list(top_negative_keywords_dict.keys())
    counts_negative = list(top_negative_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords_negative, counts_negative, color='salmon')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Negative Keywords for Customer Service')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data  # Return the negative keyword chart

def PurchaseExperiencePositiveKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "purchase_experience": {
        "positive": purchase_experience_positive_keywords,
        "negative": purchase_experience_negative_keywords
        }
    }

    results = []
    keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["purchase_experience"]["positive"]:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    top_positive_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_positive_keywords_dict = {keyword: count for keyword, count in top_positive_keywords}

    print("Top 5 Positive Keywords for Purchase Experience:", top_positive_keywords_dict)

    keywords = list(top_positive_keywords_dict.keys())
    counts = list(top_positive_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords, counts, color='skyblue')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Positive Keywords for Purchase Experience')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data

def PurchaseExperienceNegativeKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "purchase_experience": {
        "positive": purchase_experience_positive_keywords,
        "negative": purchase_experience_negative_keywords
        }
    }

    results = []
    negative_keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["purchase_experience"]["negative"]:
                negative_keyword_counts[keyword] = negative_keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    top_negative_keywords = sorted(negative_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative_keywords_dict = {keyword: count for keyword, count in top_negative_keywords}

    print("Top 5 Negative Keywords for Purchase Experience:", top_negative_keywords_dict)

    keywords_negative = list(top_negative_keywords_dict.keys())
    counts_negative = list(top_negative_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords_negative, counts_negative, color='salmon')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Negative Keywords for Purchase Experience')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    negative_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return negative_data

def PricingPositiveKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "pricing": {
        "positive": pricing_positive_keywords,
        "negative": pricing_negative_keywords
    }
    }

    results = []
    keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["pricing"]["positive"]:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    top_positive_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_positive_keywords_dict = {keyword: count for keyword, count in top_positive_keywords}

    print("Top 5 Positive Keywords for Pricing:", top_positive_keywords_dict)

    keywords = list(top_positive_keywords_dict.keys())
    counts = list(top_positive_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords, counts, color='skyblue')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Positive Keywords for Pricing')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data


def PricingNegativeKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "pricing": {
        "positive": pricing_positive_keywords,
        "negative": pricing_negative_keywords
    }
    }

    results = []
    negative_keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["pricing"]["negative"]:
                negative_keyword_counts[keyword] = negative_keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    top_negative_keywords = sorted(negative_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative_keywords_dict = {keyword: count for keyword, count in top_negative_keywords}

    print("Top 5 Negative Keywords for Pricing:", top_negative_keywords_dict)

    keywords_negative = list(top_negative_keywords_dict.keys())
    counts_negative = list(top_negative_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords_negative, counts_negative, color='salmon')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Negative Keywords for Pricing')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    negative_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return negative_data

def DeliveryPositiveKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
        "delivery": {
        "positive": delivery_positive_keywords,
        "negative": delivery_negative_keywords
    }
    }

    results = []
    keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["delivery"]["positive"]:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    top_positive_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_positive_keywords_dict = {keyword: count for keyword, count in top_positive_keywords}

    print("Top 5 Positive Keywords for Delivery:", top_positive_keywords_dict)

    keywords = list(top_positive_keywords_dict.keys())
    counts = list(top_positive_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords, counts, color='skyblue')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Positive Keywords for Delivery')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data
def DeliveryNegativeKeyword(df):
    sentiment_counts, merged_df = departments(df)

    keyword_dict = {
         "delivery": {
        "positive": delivery_positive_keywords,
        "negative": delivery_negative_keywords
    }
    }

    results = []
    negative_keyword_counts = {}

    for index, row in df.iterrows():
        review = row["review"]
        review_counts = count_keywords(review, keyword_dict)

        output_row = {"review": review}
        for keyword, data in review_counts.items():
            output_row[keyword] = data["count"]
            output_row[f"{keyword}_department"] = data["department"]

            if keyword in keyword_dict["delivery"]["negative"]:
                negative_keyword_counts[keyword] = negative_keyword_counts.get(keyword, 0) + data["count"]

        results.append(output_row)

    # Get the top 5 negative keywords based on sentiment
    top_negative_keywords = sorted(negative_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative_keywords_dict = {keyword: count for keyword, count in top_negative_keywords}

    print("Top 5 Negative Keywords for Delivery:", top_negative_keywords_dict)

    # Plotting the bar chart
    keywords_negative = list(top_negative_keywords_dict.keys())
    counts_negative = list(top_negative_keywords_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(keywords_negative, counts_negative, color='salmon')
    plt.ylabel('Keywords')
    plt.xlabel('Counts')
    plt.title('Top 5 Negative Keywords for Delivery')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    negative_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return negative_data


    # return results, top_positive_keywords_dict  # Return both the results and top keywords






def generate_sentiment_wordclouds_for_fedex(df):
    """
    Generates word clouds for positive, negative, and neutral sentiments based on the "FedEx" column.

    Args:
        df (pandas.DataFrame): DataFrame containing 'FedEx' and 'sentiment' columns.

    Returns:
        dict: A dictionary containing base64 encoded images for each sentiment (positive, negative, neutral) within the "FedEx" column.
    """

    sentiment_counts, merged_df = departments(df)
    # Filter the DataFrame for the "FedEx" column
    fedex_df = merged_df[merged_df['FedEx'] == 1]  # Assuming 'FedEx' is the value for FedEx

    # Generate word clouds for each sentiment
    # positive_words = " ".join(fedex_df[fedex_df['predicted_sentiment'] == "POSITIVE"]['review_x'])
    negative_words = " ".join(fedex_df[fedex_df['predicted_sentiment'] == "NEGATIVE"]['review_x'])
    # ... (similarly for neutral words, if applicable)

    wordcloud_width, wordcloud_height = 800, 400  # Adjust dimensions as needed

    wordclouds = {}
    for sentiment, words in [('negative', negative_words)]:
        wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height).generate(words)

        # Create a byte buffer to store the image data
        image_buffer = BytesIO()
        wordcloud.to_image().save(image_buffer, format='PNG')

        # Encode the image data as base64 for efficient transfer to HTML
        b64_encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        if sentiment in ['negative']:
            sentiment = sentiment.capitalize()

        wordclouds[sentiment] = b64_encoded_image

    return wordclouds

# def PieChartPN_plot(df): 
    
#     sentiment_counts = departments(df)
    
#     print(sentiment_counts)

#     first_dict = sentiment_counts[0]

# # Access a value from the first dictionary
#     # value1 = first_dict['key1']
    
#     positive_product_quality = first_dict["positive_product_quality"]
#     # print(positive_product_quality)
#     positive_customer_service = first_dict["positive_customer_service"]
#     positive_purchase_experience = first_dict["positive_purchase_experience"]
#     positive_pricing = first_dict["positive_pricing"]
#     positive_delivery =first_dict["positive_delivery"]
#     # postive_FedEx = first_dict["positive_FedEx"]
#     negative_product_quality = first_dict["negative_product_quality"]
#     negative_customer_service = first_dict["negative_customer_service"]
#     negative_purchase_experience = first_dict["negative_purchase_experience"]
#     negative_pricing = first_dict["negative_pricing"]
#     negative_delivery = first_dict["negative_delivery"]
#     # negative_FedEx =  first_dict["negative_FedEx"]
    

#     import matplotlib.pyplot as plt

#     # Data for positive and negative counts
#     data_positive = [
#         positive_product_quality,
#         positive_customer_service,
#         positive_purchase_experience,
#         positive_pricing,
#         positive_delivery,
#         # postive_FedEx
#     ]
#     data_negative = [
#         negative_product_quality,
#         negative_customer_service,
#         negative_purchase_experience,
#         negative_pricing,
#         negative_delivery,
#         # negative_FedEx
#     ]

#     # Labels for the pie chart
#     labels = ["Product Quality", "Customer Service", "Purchase Experience", "Pricing", "Delivery"]

#     # Colors for the pie chart slices
#     colors = ["blue", "orange", "green", "yellow", "red"]

#     # Create subplots for positive and negative pie charts
#     fig, axs = plt.subplots(1, 2, figsize=(15, 6))

#     # Plot positive pie chart
#     axs[0].pie(data_positive, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
#     axs[0].set_title("Positive Sentiments",pad=20)

# # Plot negative pie chart
#     axs[1].pie(data_negative, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
#     axs[1].set_title("Negative Sentiments",pad=20)

#     # Equal aspect ratio ensures that pie is drawn as a circle
#     axs[0].axis("equal")
#     axs[1].axis("equal")

#     plt.subplots_adjust(hspace=0.4)
#     # Show the plot
#     # plt.show()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
    
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')

#     return data
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from textwrap import wrap  # Import textwrap directly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import textwrap

# def BarChartPN_plot(df, years):
#     """
#     Creates a horizontal bar chart of positive and negative sentiment counts by department.

#     Args:
#         df (pandas.DataFrame): The input DataFrame containing the sentiment data.

#     Returns:
#         str: A base64-encoded string representing the generated plot image.
#     """

#     # Assuming departments function provides sentiment counts (modify if needed)
#     sentiment_counts = classify_reviews(df, years)
#     print('sentiment_counts', sentiment_counts)

#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#     print("Selected years:", years)

#     # Filter data for the selected years
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

#     first_dict = sentiment_counts[0]

#     # Positive and negative counts
#     positive_counts = [
#         first_dict["Positive_WebSite_Experience"],
#         first_dict["Positive_Subscription_Service"],
#         first_dict["positive_product_quality"],
#         first_dict["positive_customer_service"],
#         first_dict["positive_purchase_experience"],
#         first_dict["positive_pricing"],
#         first_dict["positive_delivery"],
#         first_dict["positive_Marketing"],
#     ]

#     negative_counts = [
#         first_dict["negative_product_quality"],
#         first_dict["negative_customer_service"],
#         first_dict["negative_purchase_experience"],
#         first_dict["negative_pricing"],
#         first_dict["negative_delivery"],
#         first_dict["Negative_Subscription_Service"],
#         first_dict["Negative_WebSite_Experience"],
#         first_dict["Negative_Marketing"],
#     ]

#     # Create labels for the departments
#     labels = [
#         "Product Quality",
#         "Delivery Service",
#         "Subscription Service",
#         "Purchase Experience",
#         "WebSite Experience",
#         "Customer Service",
#         "Product Pricing",
#         "Marketing"
#     ]

#     # Wrap text for labels
#     wrapped_labels = []
#     def wrap_text(text, max_width=14):
#         wrapped_text = textwrap.wrap(text, max_width)
#         return "\n".join(wrapped_text)
    
#     for customnam in labels:
#         wrapped_text = wrap_text(customnam)
#         wrapped_labels.append(wrapped_text)

#     # Create the chart
#     fig, ax = plt.subplots(figsize=(10,10))  # Adjust figure size for horizontal bars
#     width = 0.35

#     positive_color = 'lightgreen'
#     negative_color = 'lightcoral'

#     # Plot horizontal bars
#     y_pos = np.arange(len(labels))
#     # positive_color = plt.cm.Greens(np.linspace(0.3, 1, len(positive_counts)))

# # For negative sentiment bars (using reds)
#     # negative_color = plt.cm.Reds(np.linspace(0.3, 1, len(negative_counts)))

#     bars1 = ax.barh(y_pos - width/2, positive_counts, width, label='Positive', color=positive_color)
#     bars2 = ax.barh(y_pos + width/2, negative_counts, width, label='Negative', color=negative_color)

#     # Add counts to the right of each bar
#     for bar in bars1 + bars2:
#         width = bar.get_width()
#         ax.annotate(f'{width}',
#                     xy=(width, bar.get_y() + bar.get_height() / 2),
#                     xytext=(3, 0),  # 3 points horizontal offset
#                     textcoords="offset points",
#                     ha='left', va='center')

#     ax.set_ylabel('Departments')
#     ax.set_xlabel('Sentiment Counts')
#     ax.set_title('Sentiment Counts by Department')

#     # Set y-ticks
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(wrapped_labels)

#     # Add legend
#     ax.legend(loc='upper right', fontsize=8)

#     # Show the plots
#     plt.tight_layout()
#     plt.show()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()

#     # Encode the image data as base64
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Last thing to command out >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# def classify_reviews_BarChartPN_plot(df, years, selected_industry, config_path=r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"):
#     # Load the keyword configuration based on the selected industry
#     with open(fr"{config_path}", "r") as f: 
#         config = json.load(f)
#         print('config',config)
#         print("selected industry is ",selected_industry)
#     industry_config = config["industries"][selected_industry]["departments"]

#     print('industry_config',industry_config)


#     # Compile regex patterns for each department
#     patterns = {dept: compile_patterns(keywords) for dept, keywords in industry_config.items()}

#     # Filter DataFrame by the specified years
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

#     # Initialize the 'department' column
#     filtered_df['department'] = "Other"

#     # Classify reviews by department based on keyword matching
#     for index, row in filtered_df.iterrows():
#         if pd.notna(row['review']) and isinstance(row['review'], str):
#             review = row['review']
#             for dept, pattern in patterns.items():
#                 if pattern.search(review):
#                     filtered_df.at[index, 'department'] = dept
#                     break

#    # Initialize count dictionaries for each year
#     sentiment_counts = {}

#     # Initialize the count for each department and sentiment
#     for year in years:
#         sentiment_counts[year] = {
#             "positive": {department: 0 for department in patterns.keys()},
#             "negative": {department: 0 for department in patterns.keys()},
#         }

#     # Calculate the counts for each year, department, and sentiment
#     for index, row in filtered_df.iterrows():
#         department = row["department"]
#         year = row["RESPONSE_TIMESTAMP"].year
#         sentiment = row["predicted_sentiment"]

#         if year in sentiment_counts:
#             if sentiment == "POSITIVE":
#                 if department in sentiment_counts[year]["positive"]:
#                     sentiment_counts[year]["positive"][department] += 1
#             elif sentiment == "NEGATIVE":
#                 if department in sentiment_counts[year]["negative"]:
#                     sentiment_counts[year]["negative"][department] += 1

#     print("sentiment_countssentiment_counts",sentiment_counts)      

#     filtered_df.to_csv(r"industriesbasefile.csv", index=False)
    

#     return sentiment_counts, filtered_df

def classify_reviews_BarChartPN_plot(
    df, years, selected_industry, config_path=r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"
):
    # Load the keyword configuration based on the selected industry
    with open(fr"{config_path}", "r") as f:
        config = json.load(f)
    industry_config = config["industries"][selected_industry]["departments"]

    # Compile regex patterns for each department
    patterns = {dept: compile_patterns(keywords) for dept, keywords in industry_config.items()}

    # Filter DataFrame by the specified years
    df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

    # Initialize columns
    filtered_df['department'] = "Other"
    filtered_df['contributing_words'] = ""
    filtered_df['matched_keyword'] = ""
    filtered_df['mainword'] = ""  # New column for the main contributing word

    # Classify reviews by department and find the mainword
    for index, row in filtered_df.iterrows():
        if pd.notna(row['review']) and isinstance(row['review'], str):
            review_text = row['review']
            matched_keywords = []
            matched_departments = []

            # Find matching keywords and departments
            for department, keywords in industry_config.items():
                for keyword in keywords:
                    if keyword.lower() in review_text.lower():
                        matched_keywords.append(keyword)
                        matched_departments.append(department)

            # Assign the department with the highest keyword match count
            if matched_departments:
                most_frequent_department = max(matched_departments, key=matched_departments.count)
                filtered_df.at[index, 'department'] = most_frequent_department

            # Store contributing words and matched keywords
            filtered_df.at[index, 'contributing_words'] = ', '.join(set(matched_keywords))
            filtered_df.at[index, 'matched_keyword'] = ', '.join(matched_keywords)

            # Determine the mainword for the sentiment
            if matched_keywords:
                sentiment = row["predicted_sentiment"]
                department_keywords = industry_config.get(most_frequent_department, [])
                mainword_candidates = [kw for kw in matched_keywords if kw in department_keywords]

                if mainword_candidates:
                    # Choose the most likely keyword for the sentiment
                    filtered_df.at[index, 'mainword'] = mainword_candidates[0]
                else:
                    # Default to the first matched keyword
                    filtered_df.at[index, 'mainword'] = matched_keywords[0]

    # Initialize count dictionaries for each year
    sentiment_counts = {
        year: {
            "positive": {department: 0 for department in patterns.keys()},
            "negative": {department: 0 for department in patterns.keys()},
        }
        for year in years
    }

    # Calculate the counts for each year, department, and sentiment
    for index, row in filtered_df.iterrows():
        department = row["department"]
        year = row["RESPONSE_TIMESTAMP"].year
        sentiment = row["predicted_sentiment"]

        if year in sentiment_counts:
            if sentiment == "POSITIVE":
                if department in sentiment_counts[year]["positive"]:
                    sentiment_counts[year]["positive"][department] += 1
            elif sentiment == "NEGATIVE":
                if department in sentiment_counts[year]["negative"]:
                    sentiment_counts[year]["negative"][department] += 1

    # Save filtered DataFrame to a CSV file
    filtered_df.to_csv(r"industriesbasefile222.csv", index=False)

    return sentiment_counts, filtered_df



# def classify_reviews_BarChartPN_plot(
#     df, years, selected_industry, config_path=r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"
# ):
#     # Load the keyword configuration based on the selected industry
#     with open(fr"{config_path}", "r") as f:
#         config = json.load(f)
#         print('config', config)
#         print("selected industry is ", selected_industry)
#     industry_config = config["industries"][selected_industry]["departments"]

#     print('industry_config', industry_config)

#     # Compile regex patterns for each department
#     patterns = {dept: compile_patterns(keywords) for dept, keywords in industry_config.items()}

#     # Filter DataFrame by the specified years
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

#     # Initialize columns for department, contributing words, and matched keywords
#     filtered_df['department'] = "Other"
#     filtered_df['contributing_words'] = ""
#     filtered_df['matched_keyword'] = ""

#     # Classify reviews by department based on keyword matching
#     for index, row in filtered_df.iterrows():
#         if pd.notna(row['review']) and isinstance(row['review'], str):
#             review_text = row['review']
#             matched_keywords = []
#             matched_departments = []  # List to allow counting

#             # Find matching keywords and departments
#             for department, keywords in industry_config.items():
#                 for keyword in keywords:
#                     if keyword.lower() in review_text.lower():
#                         matched_keywords.append(keyword)
#                         matched_departments.append(department)  # Append department to list

#             # Assign the department with the highest keyword match count, if any matches are found
#             if matched_departments:
#                 most_frequent_department = max(matched_departments, key=matched_departments.count)
#                 filtered_df.at[index, 'department'] = most_frequent_department

#             # Store contributing words and matched keywords
#             filtered_df.at[index, 'contributing_words'] = ', '.join(set(matched_keywords))
#             filtered_df.at[index, 'matched_keyword'] = ', '.join(matched_keywords)

#     # Initialize count dictionaries for each year
#     sentiment_counts = {
#         year: {
#             "positive": {department: 0 for department in patterns.keys()},
#             "negative": {department: 0 for department in patterns.keys()},
#         }
#         for year in years
#     }

#     # Calculate the counts for each year, department, and sentiment
#     for index, row in filtered_df.iterrows():
#         department = row["department"]
#         year = row["RESPONSE_TIMESTAMP"].year
#         sentiment = row["predicted_sentiment"]

#         if year in sentiment_counts:
#             if sentiment == "POSITIVE":
#                 if department in sentiment_counts[year]["positive"]:
#                     sentiment_counts[year]["positive"][department] += 1
#             elif sentiment == "NEGATIVE":
#                 if department in sentiment_counts[year]["negative"]:
#                     sentiment_counts[year]["negative"][department] += 1

#     print("sentiment_counts", sentiment_counts)

#     # Save filtered DataFrame to a CSV file
#     filtered_df.to_csv(r"industriesbasefile222.csv", index=False)

#     return sentiment_counts, filtered_df






import textwrap

def wrap_text(text, max_length=15):
    """Wrap text to a specific length for better readability in the plot."""
    return '\n'.join(textwrap.wrap(text, width=max_length))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Json File  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import json

def BarChartPN_plot(df, years, selected_industry,config_path):
    """
    Creates a horizontal bar chart of positive and negative sentiment counts by department for each year.
    The number of columns in the grid layout is 3, and rows adjust based on the number of years.
    """
    # Load departments based on selected industry from JSON config
    with open(config_path,"r") as file:
        config_data = json.load(file)
    departments = list(config_data['industries'][selected_industry]['departments'].keys())

    sentiment_counts_by_year, filtered_df = classify_reviews_BarChartPN_plot(df, years, selected_industry,config_path)  # Fixed function call

    positive_color = 'lightgreen'
    negative_color = 'lightcoral'

    wrapped_departments = [textwrap.fill(dept, width=15) for dept in departments]

    # Calculate the grid layout
    num_years = len(years)
    ncols = 3
    nrows = (num_years // ncols) + (num_years % ncols > 0)

    # Create the figure and axes dynamically based on the number of years
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 9))
    axes = axes.flatten()

    positive_count = []
    negative_count = []

    # Loop through years and create individual plots
    for i, year in enumerate(years):
        ax = axes[i]
        
        # Get sentiment data for the current year
        year_sentiments = sentiment_counts_by_year.get(year, {})

        # Extract counts dynamically based on departments for the current year
        positive_counts = [year_sentiments["positive"].get(dept, 0) for dept in departments]
        negative_counts = [year_sentiments["negative"].get(dept, 0) for dept in departments]

        # Append the counts to the respective lists
        positive_count.append(positive_counts)
        negative_count.append(negative_counts)

        # Bar plots for positive and negative sentiment
        y_pos = np.arange(len(departments))
        bars1 = ax.barh(y_pos - 0.2, positive_counts, 0.4, label='Positive', color=positive_color)
        bars2 = ax.barh(y_pos + 0.2, negative_counts, 0.4, label='Negative', color=negative_color)

        # Add counts to the right of each bar
        for bar in bars1 + bars2:
            width = bar.get_width()
            ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=12)

        ax.set_ylabel('Departments', fontsize=14)
        ax.set_xlabel('Sentiment Counts', fontsize=14)
        ax.set_title(f'Sentiment Counts for {year}', fontsize=16, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_departments, fontsize=12)
        ax.legend(loc='upper right', fontsize=12)
        ax.tick_params(axis='x', labelsize=14)

    # Remove extra subplots if any (when number of years is less than grid size)
    for i in range(num_years, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    # Encode the image data as base64
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# def BarChartPN_plot(df, years,selected_industry):
#     """
#     Creates a horizontal bar chart of positive and negative sentiment counts by department for each year.
#     The number of columns in the grid layout is 3, and rows adjust based on the number of years.
#     """
#     sentiment_counts_by_year, filtered_df = classify_reviews_BarChartPN_plot(df, years,selected_industry)  # Fixed function call

#     # Plot configuration
#     labels = [
#         "Product Quality", "Delivery Service", "Subscription Service", "Purchase Experience",
#         "WebSite Experience", "Customer Service", "Product Pricing", "Marketing"
#     ]

#     # Wrap text for labels (if necessary)
#     wrapped_labels = [wrap_text(label) for label in labels]

#     positive_color = 'lightgreen'
#     negative_color = 'lightcoral'

#     # Calculate the grid layout
#     num_years = len(years)
#     ncols = 3
#     nrows = (num_years // ncols) + (num_years % ncols > 0)

#     # Create the figure and axes dynamically based on the number of years
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 8))
#     # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,16))
#     axes = axes.flatten()
    
#     positive_count = []
#     negative_count = []
#     # Loop through years and create individual plots
#     for i, year in enumerate(years):
#         ax = axes[i]
        
#         # Get sentiment data for the current year
#         year_sentiments = sentiment_counts_by_year.get(year, {})

#         print('year_sentiments',year_sentiments)

        
        
#         # Extract counts for the current year (using department keys directly)
#         # positive_counts = [
#         #     year_sentiments["positive"].get("WebSite Experience", 0),
#         #     year_sentiments["positive"].get("Subscription Service", 0),
#         #     year_sentiments["positive"].get("Product Quality", 0),
#         #     year_sentiments["positive"].get("Customer Service", 0),
#         #     year_sentiments["positive"].get("Purchase Experience", 0),
#         #     year_sentiments["positive"].get("Product Pricing", 0),
#         #     year_sentiments["positive"].get("Delivery Service", 0),
#         #     year_sentiments["positive"].get("Marketing", 0),
#         # ]
#         positive_counts = [
#             year_sentiments["positive"].get("Product Quality", 0),
#             year_sentiments["positive"].get("Delivery Service", 0),
#             year_sentiments["positive"].get("Subscription Service", 0),
#             year_sentiments["positive"].get("Purchase Experience", 0),
#             year_sentiments["positive"].get("WebSite Experience", 0),
#             year_sentiments["positive"].get("Customer Service", 0),
#             year_sentiments["positive"].get("Product Pricing", 0),
#             year_sentiments["positive"].get("Marketing", 0),
#         ]


#         # negative_counts = [
#         #     year_sentiments["negative"].get("Product Quality", 0),
#         #     year_sentiments["negative"].get("Customer Service", 0),
#         #     year_sentiments["negative"].get("Purchase Experience", 0),
#         #     year_sentiments["negative"].get("Product Pricing", 0),
#         #     year_sentiments["negative"].get("Delivery Service", 0),
#         #     year_sentiments["negative"].get("Subscription Service", 0),
#         #     year_sentiments["negative"].get("WebSite Experience", 0),
#         #     year_sentiments["negative"].get("Marketing", 0),
#         # ]
#         negative_counts = [
#             year_sentiments["negative"].get("Product Quality", 0),
#             year_sentiments["negative"].get("Delivery Service", 0),
#             year_sentiments["negative"].get("Subscription Service", 0),
#             year_sentiments["negative"].get("Purchase Experience", 0),
#             year_sentiments["negative"].get("WebSite Experience", 0),
#             year_sentiments["negative"].get("Customer Service", 0),
#             year_sentiments["negative"].get("Product Pricing", 0),
#             year_sentiments["negative"].get("Marketing", 0),
#         ]

#         # Append the counts to the respective lists
#         positive_count.append(positive_counts)
#         negative_count.append(negative_counts)



#         # Bar plots for positive and negative sentiment
#         y_pos = np.arange(len(labels))
#         bars1 = ax.barh(y_pos - 0.2, positive_counts, 0.4, label='Positive', color=positive_color)
#         bars2 = ax.barh(y_pos + 0.2, negative_counts, 0.4, label='Negative', color=negative_color)

#         # Add counts to the right of each bar
#         for bar in bars1 + bars2:
#             width = bar.get_width()
#             ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2),
#                         xytext=(3, 0), textcoords="offset points", ha='left', va='center',fontsize=12)

#         ax.set_ylabel('Departments',fontsize=14)
#         ax.set_xlabel('Sentiment Counts',fontsize=14)
#         ax.set_title(f'Sentiment Counts for {year}',fontsize=16,fontweight='bold')
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(labels,fontsize=12)
#         ax.legend(loc='upper right', fontsize=12)

#         ax.tick_params(axis='x', labelsize=14)

#     # Remove extra subplots if any (when number of years is less than grid size)
#     for i in range(num_years, len(axes)):
#         fig.delaxes(axes[i])



#     print("positive_count",positive_count)
#     print("negative_count",negative_count)

#     plt.tight_layout()
#     plt.show()

#     # Save the plot to a buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()

#     # Encode the image data as base64
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data



# def BarChartPN_plot(df, years):
#     """
#     Creates a horizontal bar chart of positive and negative sentiment counts by department.
#     The number of columns in the grid layout is 3, and rows adjust based on the number of years.

#     Args:
#         df (pandas.DataFrame): The input DataFrame containing the sentiment data.
#         years (list): A list of years selected for plotting.

#     Returns:
#         str: A base64-encoded string representing the generated plot image.
#     """

#     # Assuming departments function provides sentiment counts (modify if needed)
#     sentiment_counts = classify_reviews(df, years)
#     print('sentiment_counts', sentiment_counts)

#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#     print("Selected years:", years)

#     # Filter data for the selected years
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

#     # Create the chart
#     labels = [
#         "Product Quality",
#         "Delivery Service",
#         "Subscription Service",
#         "Purchase Experience",
#         "WebSite Experience",
#         "Customer Service",
#         "Product Pricing",
#         "Marketing"
#     ]

#     # Wrap text for labels
#     wrapped_labels = []
#     def wrap_text(text, max_width=14):
#         wrapped_text = textwrap.wrap(text, max_width)
#         return "\n".join(wrapped_text)
    
#     for customnam in labels:
#         wrapped_text = wrap_text(customnam)
#         wrapped_labels.append(wrapped_text)

#     # Plot configuration
#     positive_color = 'lightgreen'
#     negative_color = 'lightcoral'

#     # Calculate the grid layout
#     num_years = len(years)
#     ncols = 3
#     nrows = (num_years // ncols) + (num_years % ncols > 0)

#     # Create the figure and axes dynamically based on the number of years
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 6))
#     axes = axes.flatten()  # Flatten in case there are extra rows or columns

#     # Loop through years and create individual plots
#     for i, year in enumerate(years):
#         ax = axes[i]

#         first_dict = sentiment_counts[i]  # Sentiment data for the year

#         # Positive and negative counts
#         positive_counts = [
#             first_dict["Positive_WebSite_Experience"],
#             first_dict["Positive_Subscription_Service"],
#             first_dict["positive_product_quality"],
#             first_dict["positive_customer_service"],
#             first_dict["positive_purchase_experience"],
#             first_dict["positive_pricing"],
#             first_dict["positive_delivery"],
#             first_dict["positive_Marketing"],
#         ]

#         negative_counts = [
#             first_dict["negative_product_quality"],
#             first_dict["negative_customer_service"],
#             first_dict["negative_purchase_experience"],
#             first_dict["negative_pricing"],
#             first_dict["negative_delivery"],
#             first_dict["Negative_Subscription_Service"],
#             first_dict["Negative_WebSite_Experience"],
#             first_dict["Negative_Marketing"],
#         ]

#         # Bar plots for positive and negative sentiment
#         y_pos = np.arange(len(labels))
#         bars1 = ax.barh(y_pos - 0.2, positive_counts, 0.4, label='Positive', color=positive_color)
#         bars2 = ax.barh(y_pos + 0.2, negative_counts, 0.4, label='Negative', color=negative_color)

#         # Add counts to the right of each bar
#         for bar in bars1 + bars2:
#             width = bar.get_width()
#             ax.annotate(f'{width}',
#                         xy=(width, bar.get_y() + bar.get_height() / 2),
#                         xytext=(3, 0),  # 3 points horizontal offset
#                         textcoords="offset points",
#                         ha='left', va='center')

#         ax.set_ylabel('Departments')
#         ax.set_xlabel('Sentiment Counts')
#         ax.set_title(f'Sentiment Counts for {year}')

#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(wrapped_labels)
#         ax.legend(loc='upper right', fontsize=8)

#     # Remove extra subplots if any (when number of years is less than grid size)
#     for i in range(num_years, len(axes)):
#         fig.delaxes(axes[i])

#     plt.tight_layout()
#     plt.show()

#     # Save the plot to a buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()

#     # Encode the image data as base64
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data







# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Cluster base df code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# def BarChartPN_plot():
#     """
#     Creates a side-by-side bar chart of positive and negative sentiment counts by department.

#     Args:
#         df (pandas.DataFrame): The input DataFrame containing the sentiment data.

#     Returns:
#         str: A base64-encoded string representing the generated plot image.
#     """


#     df = pd.read_csv(r'C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\customer_segments.csv')

#     # Ensure required columns exist
#     if 'predicted_sentiment' not in df.columns or 'Customer_Segment' not in df.columns:
#         raise ValueError("DataFrame must contain 'predicted_sentiment' and 'Customer_segment' columns.")

#     # Count sentiments by department
#     sentiment_counts = df.groupby(['Customer_Segment', 'predicted_sentiment']).size().unstack(fill_value=0)

#     # Get counts for each sentiment
#     positive_counts = sentiment_counts.get('POSITIVE', 0).values
#     negative_counts = sentiment_counts.get('NEGATIVE', 0).values

#     # Create labels for the departments
#     labels = sentiment_counts.index.tolist()

#     # Wrap labels for better readability
#     wrapped_labels = [textwrap.fill(label, width=15) for label in labels]

#     # Create the chart
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     # Adjust width to prevent overlapping bars
#     width = 0.35

#     positive_color = 'lightgreen'
#     negative_color = 'lightcoral'

#     # Plot the bars
#     bars1 = ax.bar(np.arange(len(labels)) - width/2, positive_counts, width, label='Positive', color=positive_color)
#     bars2 = ax.bar(np.arange(len(labels)) + width/2, negative_counts, width, label='Negative', color=negative_color)

#     # Add counts above each bar
#     for bar in bars1 + bars2:
#         height = bar.get_height()
#         ax.annotate(f'{height}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 2),  # 2 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

#     ax.set_xlabel('Departments')
#     ax.set_ylabel('Sentiment Counts')
#     ax.set_title('Sentiment Counts by Department')

#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(wrapped_labels, rotation=0)

#     # Add legend
#     ax.legend(loc='upper right', fontsize=8)

#     # plt.show()

#     # Save the plot to a buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()

#     # Encode the image data as base64
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>






# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NPS Score  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def NPSScore(df, years):
    # Filter DataFrame for the selected years
    df = convert_timestamp(df.copy())
    
    print("Selected years:", years)

    # Filter data for the selected years
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

    print(filtered_df.head(10))

    # Count reviews based on predicted sentiment
    positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'POSITIVE'].shape[0]
    negative_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'NEGATIVE'].shape[0]
    neutral_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'NEUTRAL'].shape[0]

    print('positive_reviews_NPSScore:', positive_reviews)
    print('negative_reviews_NPSScore:', negative_reviews)
    print('neutral_reviews_NPSScore:', neutral_reviews)

    total_reviews = positive_reviews + negative_reviews + neutral_reviews

    # Handle case where there are no reviews
    if total_reviews == 0:
        print("No reviews to analyze.")
        return {'NPSSS': None, 'NPSScoress': None}  # Return None for both if no reviews

    # Calculate percentages
    positive_percentage = (positive_reviews / total_reviews) * 100
    negative_percentage = (negative_reviews / total_reviews) * 100
    neutral_percentage = (neutral_reviews / total_reviews) * 100

    # Calculate NPS
    NPS = positive_percentage - negative_percentage

    # Create the pie chart data
    sizes = [negative_reviews, neutral_reviews, positive_reviews]
    labels = ['Detractors', 'Passives', 'Promoters']
    colors = ['red', 'orange', 'green']

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(4,5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct='%1.1f%%',
        pctdistance=0.80,
        # radius=0.3,  
        textprops={'fontsize': 10}
    )

    # Change percentage color to white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Draw a circle at the center to create a "whole" effect
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)

    # Set the NPS value in the center
    ax.annotate(f'NPS: {round(NPS)}',
                xy=(0, 0),
                fontsize=16,
                fontweight='bold',
                color='black',
                ha='center',
                va='center')

    # Set equal aspect ratio and tighten layout
    ax.axis('equal')
    plt.tight_layout()
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data



# <<<<<<<<<<<<<<<<<<<<<<  NPS  >>>>>>>>>>>>>>>>>>>>>>>>>>


# def NetPro(df):

#     import numpy as np

#     # filtered_df = df[(df['review'] != "undefined") ] 
#     # df['nps_new'] = np.where(df['predicted_sentiment'] == 'POSITIVE', 'Promoter', np.where(df['predicted_sentiment'] == 'NEGATIVE', 'Detractor',np.where(df['predicted_sentiment'] == 'NEUTRAL','Passive')))

#     df['nps_new'] = np.where(df['predicted_sentiment'] == 'POSITIVE', 'Promoter',
#                          np.where(df['predicted_sentiment'] == 'NEGATIVE', 'Detractor',
#                                   np.where(df['predicted_sentiment'] == 'NEUTRAL', 'Passive',
#                                           'Other')))


#     promoters = df[df['nps_new'] == 'Promoter'].shape[0]
#     passives = df[df['nps_new'] == 'Passive'].shape[0]
#     detractors = df[df['nps_new'] == 'Detractor'].shape[0]
#     total_reviews = promoters + passives + detractors

#     # Define colors for each category
#     colors = ['green', 'orange', 'red']

#     # Create subplots for the pie charts
#     fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 3 columns

#     # Create pie charts for each review category
#     reviews_data = {'Promoters': promoters, 'Passive': passives, 'Detractor': detractors}

#     for ax, (label, count), color in zip(axs, reviews_data.items(), colors):
#         sizes = [count, total_reviews - count]  # Current vs Remaining

#         # Create pie chart with customized labels and autopct function
#         wedges, texts = ax.pie(
#             sizes,
#             labels=[label,''],
#             # labels=[],
#             colors=[color, 'white'],
#             startangle=90,
#             # autopct=lambda p: f'{int(p)}%' if p == (count / total_reviews) * 100 else '',
#             pctdistance=0.85,
#             textprops={'fontsize': 11}
#         )

#         # Set edge color for each wedge and adjust linewidth
#         for wedge in wedges:
#             wedge.set_edgecolor(color)
#             wedge.set_linewidth(3)

#         # Draw a circle at the center for a "whole" effect
#         centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor=color, linewidth=2)
#         ax.add_artist(centre_circle)

#         # Add percentage below the label for current reviews
#         ax.annotate(f'{(count / total_reviews) * 100:.1f}%',
#                     xy=(-0.1, 0.4),
#                     fontsize=11,
#                     color=color,
#                     ha='center')

#     # Set equal aspect ratio for all pie charts
#     for ax in axs:
#         ax.axis('equal')

#     # Set the title for the entire figure
#     plt.suptitle('Net Promoter Score (NPS)', fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title
#     # plt.show()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data


import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def NetPro(df, years):
    # Filter DataFrame for the selected years
    df = convert_timestamp(df.copy())
    
    print("Selected years:", years)

    # Filter data for the selected years
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

    # Categorize the predicted sentiments
    filtered_df['nps_new'] = np.where(filtered_df['predicted_sentiment'] == 'POSITIVE', 'Promoter',
                             np.where(filtered_df['predicted_sentiment'] == 'NEGATIVE', 'Detractor',
                                      np.where(filtered_df['predicted_sentiment'] == 'NEUTRAL', 'Passive',
                                              'Other')))

    promoters = filtered_df[filtered_df['nps_new'] == 'Promoter'].shape[0]
    passives = filtered_df[filtered_df['nps_new'] == 'Passive'].shape[0]
    detractors = filtered_df[filtered_df['nps_new'] == 'Detractor'].shape[0]
    total_reviews = promoters + passives + detractors

    # Handle case where there are no reviews
    if total_reviews == 0:
        print("No reviews to analyze.")
        return None  # Return None or handle accordingly if no reviews

    # Define colors for each category
    colors = ['green', 'orange', 'red']

    # Create subplots for the pie charts
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 3 columns

    # Create pie charts for each review category
    reviews_data = {'Promoters': promoters, 'Passive': passives, 'Detractor': detractors}

    for ax, (label, count), color in zip(axs, reviews_data.items(), colors):
        sizes = [count, total_reviews - count]  # Current vs Remaining

        # Create pie chart with customized labels and autopct function
        wedges, texts = ax.pie(
            sizes,
            labels=[label, ''],
            colors=[color, 'white'],
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 11}
        )

        # Set edge color for each wedge and adjust linewidth
        for wedge in wedges:
            wedge.set_edgecolor(color)
            wedge.set_linewidth(3)

        # Draw a circle at the center for a "whole" effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor=color, linewidth=2)
        ax.add_artist(centre_circle)

        # Add percentage below the label for current reviews
        ax.annotate(f'{(count / total_reviews) * 100:.1f}%',
                    xy=(-0.1, 0.4),
                    fontsize=11,
                    color=color,
                    ha='center')

    # Set equal aspect ratio for all pie charts
    for ax in axs:
        ax.axis('equal')

    # Set the title for the entire figure
    plt.suptitle('Net Promoter Score (NPS)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title

    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode the image for response
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data













# #<<<<<<<<<<<<<<<<<<<<<    Count_of keywords in each department(Bar Graph) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import matplotlib.pyplot as plt
import io
import base64

import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator  # Import for wrapping text

def Count_of_keywords_in_each_department(df):

    sentiment_counts, merged_df = departments(df)

    # Count occurrences of each department and merge with sentiment counts
    department_counts = merged_df[["product_quality", "customer_service", "purchase_experience", "pricing", "delivery"]].sum()
    # department_counts = pd.concat([department_counts, sentiment_counts[0]], axis=1)

    result_dict = {
    'Department': ["product_quality", "customer_service", "purchase_experience", "pricing", "delivery"],
    'Positive_Reviews': department_counts.values.tolist()
        }
    df = pd.DataFrame(result_dict)

    print(result_dict)

    # Get the top 5 departments by positive review count
    top_departments = df.nlargest(5, 'Positive_Reviews')

    # Define colors for the bars
    colors = ['lightgrey'] * len(top_departments)

    # Highlight the maximum and minimum values
    
    max_value_index = top_departments['Positive_Reviews'].idxmax()
    min_value_index = top_departments['Positive_Reviews'].idxmin()

    print('max_value_index',max_value_index)
    print('min_value_index',min_value_index)

    custom_names = {
        'product_quality': 'Product Quality',
        'customer_service': 'Customer Service',  # Example of wrapping
        'purchase_experience': 'Purchase Experience',
        'pricing': 'Pricing',
        'delivery': 'Delivery'
    }
    import textwrap
    # Wrap text on x-axis
    def wrap_text(text, max_width=15):
        return "\n".join(textwrap.wrap(text, max_width))

    wrapped_supplier_names = [wrap_text(custom_names[name]) for name in department_counts.index]
        
        
    # top_department_names = top_departments.index.map(custom_names).tolist()


    colors[top_departments.index.get_loc(max_value_index)] = 'lightgreen'  # Highlight max with light green
    colors[top_departments.index.get_loc(min_value_index)] = 'lightcoral'  # Highlight min with light red

    # Plotting
    plt.figure(figsize=(6, 5))
    bars = plt.bar(top_departments['Department'], top_departments['Positive_Reviews'], color=colors)
    

    # Customize the chart
    plt.xlabel('Department')
    plt.ylabel('Positive Review Count')
    plt.title('Top 5 Departments by Positive Review Count')
    # plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    import textwrap
  # Wrap text on x-axis
    def wrap_text(text, max_width=15):
      return "\n".join(textwrap.wrap(text, max_width))

    wrapped_supplier_names = [wrap_text(custom_names[name]) for name in department_counts.index]

  # Set wrapped names as x-axis labels
    plt.xticks(department_counts.index, wrapped_supplier_names, rotation=0)



    # Display data values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10, color='black')

    # Save the plot to a BytesIO object (if needed)
    # ... (code for saving and encoding the image can be added here)

    # Show the plot
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')

    # Encode the image data as base64
    image_data = b64encode(buffer.getvalue()).decode('utf-8')

    # Close the buffer (optional)
    buffer.close()

    # Clear the plot (optional to avoid memory leaks)
    plt.clf()

    # Return the image data
    return image_data
    # plt.show()


# def Count_of_keywords_in_each_department(df):
  

#   # Assuming departments function provides sentiment counts and merged DataFrame (modify if needed)
#   sentiment_counts, merged_df = departments(df)

#   # Count occurrences of each department
#   department_counts = merged_df[["product_quality", "customer_service", "purchase_experience", "pricing", "delivery"]].sum()

#   # Customize bar names and colors (modify as needed)
#   custom_names = {
#       "product_quality": "Product Quality",
#       "customer_service": "Customer Service",
#       "purchase_experience": "Purchase Experience",
#       "pricing": "Pricing",
#       "delivery": "Delivery",
#   }
#   custom_colors = ['green', 'grey', 'grey', 'grey', 'red']  # Modify color list

#   # Create a bar chart in memory
#   plt.figure(figsize=(10, 6))
#   bars = plt.bar(department_counts.index, department_counts.values, color=custom_colors)  # Use custom colors

#   # Set customized bar labels and title
#   plt.xlabel("Department")
#   plt.ylabel("Keyword Count")
#   plt.title("Keyword Counts in Each Department")

#   import textwrap
#   # Wrap text on x-axis
#   def wrap_text(text, max_width=15):
#       return "\n".join(textwrap.wrap(text, max_width))

#   wrapped_supplier_names = [wrap_text(custom_names[name]) for name in department_counts.index]

#   # Set wrapped names as x-axis labels
#   plt.xticks(department_counts.index, wrapped_supplier_names, rotation=0)  # Use custom bar names and positions

#   # Add values above each bar
#   for bar, value in zip(bars, department_counts):
#       yval = value + 1  # Adjust y-offset if needed
#       plt.text(bar.get_x() + bar.get_width() / 2, yval, int(value), ha='center', va='bottom')

#   # Save the plot to a BytesIO object
#   buffer = io.BytesIO()
#   plt.savefig(buffer, format='png')

#   # Encode the image data as base64
#   image_data = b64encode(buffer.getvalue()).decode('utf-8')

#   # Close the buffer (optional)
#   buffer.close()

#   # Clear the plot (optional to avoid memory leaks)
#   plt.clf()

#   # Return the image data
#   return image_data

import matplotlib.pyplot as plt
import io
import base64

# def Count_of_keywords_in_each_department(df):
  
#   sentiment_counts, merged_df = departments(df)  # Assuming departments function exists

#   # Count occurrences of each department
#   department_counts = merged_df[["product_quality", "customer_service", "purchase_experience", "pricing", "delivery"]].sum()

#   print(merged_df.head(50))
  
#   color_map = {
#     "product_quality": "green",
#     "customer_service": "black",
#     "purchase_experience": "blue",
#     "pricing": "orange",
#     "delivery": "red"
#     }

#   bars = plt.bar(department_counts.index, department_counts.values, color=[color_map[dept] for dept in department_counts.index])


#   plt.figure(figsize=(10, 6))

#   for bar, count in zip(bars, department_counts):
#     yval = bar.get_height()  # Get the height of each bar
#     plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, str(int(count)), ha='center', va='bottom')  # Adjust offset for better positioning

#   plt.xlabel("Department")
#   plt.ylabel("Keyword Count")
#   plt.title("Keyword Counts in Each Department",fontweight='bold', fontsize = 14,fontfamily='Calibri')
#   custom_labels = ["Product Quality", "Customer Service", "Purchase Experience", "Pricing", "Delivery"]

#   import textwrap
  
#   def wrap_text(text, max_width=15):
    
#     return "\n".join(textwrap.wrap(text, max_width))
    
#   wrapped_labels = [wrap_text(label) for label in custom_labels]

#   plt.xticks(range(len(department_counts.index)), wrapped_labels, rotation=0, ha='right',fontweight='bold',fontsize = 8,fontfamily='Calibri')

#   # Save the plot to a BytesIO object
#   buffer = io.BytesIO()
#   plt.savefig(buffer, format='png')

#   # Encode the image data as base64
#   image_data = b64encode(buffer.getvalue()).decode('utf-8')

#   # Close the buffer (optional)
#   buffer.close()

#   # Clear the plot (optional to avoid memory leaks)
#   plt.clf()

#   # Return the image data
#   return image_data





def create_plot(df):
    
    # global excel_file
    global top_5_suppliers

    # excel_file = (r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv')

    

    # Assuming your DataFrame is named 'df' and has columns 'supplier_name' and 'predicted_sentiment'

    # Filter non-undefined reviews
    # df = pd.read_csv(excel_file)

    filtered_df = df[(df['predicted_sentiment'] != "NEUTRAL") & (df['review'] != "undefined") & (df['supplier_name'] != "undefined")]  

    filtered_df.to_csv('filtered_data.csv', index=False)


    # Filter positive sentiment reviews
    positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == "POSITIVE"]

    # Group by supplier name and count positive reviews
    positive_reviews_by_supplier = positive_reviews.groupby('supplier_name').size()

    # Sort by count in descending order (most positive to least)
    top_5_suppliers = positive_reviews_by_supplier.sort_values(ascending=False).head(5)

    # print("top_5_suppliers",top_5_suppliers)
    supplier_names = top_5_suppliers.index

    
    # Check if any top 5 have zero reviews (indicating no positive reviews)
    if 0 in top_5_suppliers.values:
        # Filter out suppliers with zero reviews (optional)
        top_5_suppliers = top_5_suppliers[top_5_suppliers != 0]

    
    plt.switch_backend('Agg')  # Ensures compatibility with Django rendering

    plt.figure(figsize=(10, 4))

    # Set bar color to light green
    bar_color = 'lightgreen'
    top_5_suppliers = top_5_suppliers[::-1]
    # Create the bar chart
    bars = plt.barh(top_5_suppliers.index, top_5_suppliers.values, color=bar_color)

    for i, (value, name) in enumerate(zip(top_5_suppliers.values, supplier_names)):
        plt.text(value + 0.2, i, str(value), ha='left', va='center', fontsize=10,)  # Adjust offset and font size

    # plt.gca().invert_xaxis()

    print("top_5_suppliers.values",top_5_suppliers.values)
    # Calculate total for percentage calculation
    total_positive_reviews = top_5_suppliers.sum()
    print("total_positive_reviews",total_positive_reviews)

   
    plt.xlabel('')
    # plt.ylabel('Number of Positive Reviews', fontsize=16, fontweight='bold', fontfamily='Calibri')
    plt.title('*****Top 5 Suppliers by Positive Review', fontsize=20, fontweight='bold')
    plt.ylabel('Number of Positive Reviews', fontsize=16, fontweight='bold')
    # plt.title('Top 5 Suppliers by Positive Review', fontsize=16, fontweight='bold', fontfamily='Calibri')  
    # plt.title('*****Top 5 Suppliers by Positive Review', fontsize=20, fontweight='bold')  

    # plt.yticks(fontweight='bold', fontsize = 12)
    # plt.xticks()

    # plt.xticks(rotation=45)
    # plt.xticks(rotation=45, fontsize=10,fontweight='bold')
    import textwrap

    def wrap_text(text, max_width=15):
        return "\n".join(textwrap.wrap(text, max_width))

    wrapped_supplier_names = [wrap_text(name) for name in supplier_names]

    # Set the wrapped names as x-axis labels
    # plt.yticks(range(len(supplier_names)), wrapped_supplier_names,fontname='Verdana')

    # plt.xticks(fontname='Verdana')

    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

    plt.tight_layout() 
    # plt.show()

    # Convert the plot to a base64-encoded image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    total_positive_reviews = top_5_suppliers.sum()

  # Create the table data as a list of lists
    table_data = []
    for name, value in top_5_suppliers.items():
      percentage = value / total_positive_reviews * 100
      table_data.append([name, f"{percentage:.1f}%"])

    # Wrap supplier names for table (optional)
    wrapped_supplier_names = [wrap_text(name) for name in top_5_suppliers.index]


    return data



def get_bottom_5_negative_suppliers(df):

    # global excel_file
    global top_5_suppliers


    # excel_file = (r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv')

    # df = pd.read_csv(excel_file)
    filtered_df = df[(df['predicted_sentiment'] != "NEUTRAL") & (df['review'] != "undefined") & (df['supplier_name'] != "undefined")] 

    # Filter negative sentiment reviews
    negative_reviews = filtered_df[filtered_df['predicted_sentiment'] == "NEGATIVE"]

    # Handle cases with no negative reviews:
    if negative_reviews.empty:
        print("No negative reviews found. Returning None.")
        return None

    # Group by supplier name and count negative reviews
    negative_reviews_by_supplier = negative_reviews.groupby('supplier_name').size()

    # Sort by count in descending order (most negative to least)
    bottom_5_suppliers = negative_reviews_by_supplier.sort_values(ascending=False).head(5)


    print(bottom_5_suppliers)

    supplier_names = bottom_5_suppliers.index


    # Create the bar chart
    plt.switch_backend('Agg')  # Ensures compatibility with Django rendering

    plt.figure(figsize=(10, 4))

    # Set bar color to light red
    bar_color = 'lightcoral'

    # Create the bars
    # bars = plt.bar(bottom_5_suppliers.index, bottom_5_suppliers.values, color=bar_color)
    bottom_5_suppliers = bottom_5_suppliers[::-1]
    bars = plt.barh(bottom_5_suppliers.index, bottom_5_suppliers.values, color=bar_color)

    for i, (value, name) in enumerate(zip(bottom_5_suppliers.values, supplier_names)):
        plt.text(value + 0.1, i, str(value), ha='left', va='center', fontsize=10)


    # Calculate total negative reviews for percentage calculation
    total_negative_reviews = bottom_5_suppliers.sum()

    # ... (rest of your code)

    # Add percentage labels above each bar
    # for bar, value in zip(bars, bottom_5_suppliers.values):
    #     label_x = bar.get_bbox().xmax + 5  # Adjust spacing as needed
    #     label_y = bar.get_y() + bar.get_height() / 2  # Center label vertically
    #     plt.text(label_x, label_y, f"{value}", ha='left', va='center', fontsize=10)

    # ... (rest of your code)


    plt.xlabel('')
    plt.ylabel('Supplier Name',fontsize=16, fontweight='bold', fontfamily='Calibri',rotation = 90)
    plt.title('Top 5 Suppliers by Negative Review',fontsize=16, fontweight='bold', fontfamily='Calibri')
    # plt.xticks(rotation=45)
    plt.yticks(fontweight='bold', fontsize = 12,fontname='Calibri')
    # plt.xticks(rotation=90, ha='right') 
    # plt.xticks(rotation=45, fontsize=12,fontweight='bold')
    import textwrap

    def wrap_text(text, max_width=15):
        return "\n".join(textwrap.wrap(text, max_width))

    wrapped_supplier_names = [wrap_text(name) for name in supplier_names]

    # Set the wrapped names as x-axis labels
    plt.yticks(range(len(supplier_names)), wrapped_supplier_names,fontfamily='Calibri')
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

    plt.tight_layout() 

    # plt.show()
    # Convert the plot to a base64-encoded image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return data

def create_positive_sentiment_plot(df):

#   global excel_file

#   excel_file = (r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv')

  # Assuming your DataFrame is named 'df' and has columns 'product_name' and 'predicted_sentiment'

#   df = pd.read_csv(excel_file)

  # Filter non-neutral reviews and positive sentiment
  filtered_df = df[(df['predicted_sentiment'] != "NEUTRAL") & (df['review'] != "undefined") & (df['product_fullname'] != "undefined")] 
  positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == "POSITIVE"]

  # Group by product name and count positive reviews
  positive_reviews_by_product = positive_reviews.groupby('product_fullname').size()

  # Sort by count in descending order (most positive to least)
  top_5_positive = positive_reviews_by_product.sort_values(ascending=False).head(5)

  supplier_names = top_5_positive.index

  # Create the bar chart
  plt.switch_backend('Agg')  # Ensures compatibility with Django rendering

  plt.figure(figsize=(10, 4))

  # Set bar color to light green
  bar_color = 'lightgreen'

  # Create the bars

  top_5_positive = top_5_positive[::-1]

  bars = plt.barh(top_5_positive.index, top_5_positive.values, color=bar_color)

  for i, (value, name) in enumerate(zip(top_5_positive.values, supplier_names)):
        plt.text(value + 0.1, i, str(value), ha='left', va='center', fontsize=10)

  # Calculate total for percentage calculation
  total_positive_reviews = top_5_positive.sum()

  # Add percentage labels above each bar
#   for bar, value in zip(bars, top_5_positive.values):
#       yval = value / total_positive_reviews * 100  # Calculate percentage
#       # Adjust positioning to avoid overlapping bars and labels:
#       label_y = yval + bar.get_height() * 0.1 # Place label above bar with some vertical space
#       plt.text(bar.get_x() + bar.get_width() / 2, label_y, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)  # Adjust font size for readability

  plt.xlabel('')
  plt.ylabel('Number of Positive Reviews',fontsize=16, fontweight='bold', fontfamily='Calibri')
  plt.title('Top 5 Products by Positive Reviews',fontsize=16, fontweight='bold', fontfamily='Calibri')
  plt.yticks(fontweight='bold', fontsize = 12)
#   plt.xticks(rotation=45)

#   plt.xticks(rotation=90, ha='right') 
#   plt.xticks(rotation=45, fontsize=12,fontweight='bold')
  import textwrap
  def wrap_text(text, max_width=15):
    return "\n".join(textwrap.wrap(text, max_width))

  wrapped_supplier_names = [wrap_text(name) for name in supplier_names]

    # Set the wrapped names as x-axis labels
  plt.yticks(range(len(supplier_names)), wrapped_supplier_names,fontfamily='Calibri')

  plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

  plt.tight_layout() 

  # Convert the plot to a base64-encoded image
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close()
  data = base64.b64encode(buf.getvalue()).decode('utf-8')

  return data


def create_negative_sentiment_plot(df):
  
  """
  Finds the bottom 5 suppliers with the most negative sentiment reviews.

  Args:
      excel_file (str): Path to the CSV file containing sentiment data.

  Returns:
      str: Base64-encoded PNG image of the bar chart or None if no negative reviews exist.
  """

  # Read data and filter non-undefined reviews (assuming "NEUTRAL" signifies undefined)
#   df = pd.read_csv(excel_file)
  filtered_df =   filtered_df = df[(df['predicted_sentiment'] != "NEUTRAL") & (df['review'] != "undefined") & (df['product_fullname'] != "undefined")] 


  # Filter negative sentiment reviews
  negative_reviews = filtered_df[filtered_df['predicted_sentiment'] == "NEGATIVE"]

  # Handle cases with no negative reviews:
  if negative_reviews.empty:
    print("No negative reviews found. Returning None.")
    return None

  # Group by supplier name and count negative reviews
  negative_reviews_by_supplier = negative_reviews.groupby('product_fullname').size()

  # Sort by count in descending order (most negative to least)
  bottom_5_suppliers = negative_reviews_by_supplier.sort_values(ascending=False).head(5)

  print("bottom_5_suppliers",bottom_5_suppliers)

  supplier_names = bottom_5_suppliers.index

  # Create the bar chart
  plt.switch_backend('Agg')  # Ensures compatibility with Django rendering

  plt.figure(figsize=(10, 4))

  # Set bar color to light red
  bar_color = 'lightcoral'

  bottom_5_suppliers = bottom_5_suppliers[::-1]
  # Create the bars
  bars = plt.barh(bottom_5_suppliers.index, bottom_5_suppliers.values, color=bar_color)
  for i, (value, name) in enumerate(zip(bottom_5_suppliers.values, supplier_names)):
        plt.text(value + 0.1, i, str(value), ha='left', va='center', fontsize=10)

  # Calculate total negative reviews for percentage calculation
  total_negative_reviews = bottom_5_suppliers.sum()

  print("total_negative_reviews",total_negative_reviews)

  # Add percentage labels above each bar
#   for bar, value in zip(bars, bottom_5_suppliers.values):
#     yval = value / total_negative_reviews * 100
#     label_y = yval + bar.get_height() * 0.1  # Place label above bar with some vertical space
#     plt.text(bar.get_x() + bar.get_width() / 2, label_y, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

  plt.xlabel('')
#   plt.rc('axes', labelsize=18)
  plt.ylabel('Product Name',fontsize=16, fontweight='bold', fontfamily='Calibri')
  plt.title('Top 5 Products by Negative Review',fontsize=16, fontweight='bold', fontfamily='Calibri')
  plt.yticks(fontweight='bold', fontsize = 12)
#   plt.xticks(rotation=45)
#   plt.xticks(rotation=45, fontsize=12,fontweight='bold')
  import textwrap
  def wrap_text(text, max_width=15):
    return "\n".join(textwrap.wrap(text, max_width))

  wrapped_supplier_names = [wrap_text(name) for name in supplier_names]

    # Set the wrapped names as x-axis labels
  plt.yticks(range(len(supplier_names)), wrapped_supplier_names,fontfamily='Calibri')
  plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

  plt.tight_layout() 


  # Convert the plot to a base64-encoded image
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close()
  data = base64.b64encode(buf.getvalue()).decode('utf-8')

  return data


# def generate_sentiment_wordclouds():
#   """
#   Generates word clouds for positive, negative, and neutral sentiments.

#   Args:
#       df (pandas.DataFrame): DataFrame containing 'review' and 'predicted_sentiment' columns.

#   Returns:
#       dict: A dictionary containing base64 encoded images for each sentiment (positive, negative, neutral).
#   """
#   df = pd.read_csv(excel_file)

#   positive_words = " ".join(df[df['predicted_sentiment'] == 'POSITIVE']['review'])
#   negative_words = " ".join(df[df['predicted_sentiment'] == 'NEGATIVE']['review'])
# #   neutral_words = " ".join(df[df['predicted_sentiment'] == 'NEUTRAL']['review'])

#   wordcloud_width, wordcloud_height = 800, 400  # Adjust dimensions as needed

#   wordclouds = {}
#   for sentiment, words in [('negative', negative_words),('positive', positive_words)]:
#     wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height).generate(words)

#     # Create a byte buffer to store the image data
#     image_buffer = BytesIO()
#     wordcloud.to_image().save(image_buffer, format='PNG')

#     # Encode the image data as base64 for efficient transfer to HTML
#     b64_encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

#     if sentiment == 'positive' or sentiment == 'negative':
#             sentiment = sentiment.capitalize()

#     wordclouds[sentiment] = b64_encoded_image

#   return wordclouds

def generate_sentiment_wordclouds(df):
    """
    Generates word clouds for positive, negative, and neutral sentiments.

    Args:
        df (pandas.DataFrame): DataFrame containing 'review' and 'predicted_sentiment' columns.

    Returns:
        dict: A dictionary containing base64 encoded images for each sentiment (positive, negative, neutral).
    """
    # df = pd.read_csv(excel_file)
    positive_words = " ".join(df[df['predicted_sentiment'] == 'POSITIVE']['review'])
    negative_words = " ".join(df[df['predicted_sentiment'] == 'NEGATIVE']['review'])
    # neutral_words = " ".join(df[df['predicted_sentiment'] == 'NEUTRAL']['review'])

    wordcloud_width, wordcloud_height = 800, 400  # Adjust dimensions as needed

    wordclouds = {}
    for sentiment, words in [('positive', positive_words), ('negative', negative_words)]:
        wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height).generate(words)

        # Create a byte buffer to store the image data
        image_buffer = BytesIO()
        wordcloud.to_image().save(image_buffer, format='PNG')

        # Encode the image data as base64 for efficient transfer to HTML
        b64_encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        if sentiment in ['positive', 'negative']:
            sentiment = sentiment.capitalize()


        wordclouds[sentiment] = b64_encoded_image

    return wordclouds



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Trend line >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from io import BytesIO
import base64

# Read the CSV file
# df = pd.read_csv(r'/content/YES.csv')  # Assuming the file is uploaded to Colab

# # Ensure RESPONSE_TIMESTAMP is datetime format
# def convert_timestamp(df):
#   df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#   return df

# df = convert_timestamp(df.copy())  # Apply conversion to a copy of df

# # Define year options for the dropdown
# year_options = df['RESPONSE_TIMESTAMP'].dt.year.unique()

# # Create the dropdown widget
# year_dropdown = widgets.Dropdown(
#     options=year_options,
#     description="Year:",
#     value=year_options.min()  # Set initial value to the minimum year
# )

# def update_plot(year):
#   # Filter data for the selected year
#   filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year == year]

#   # Count sentiment values for each month
#   monthly_counts = filtered_df.groupby(pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'))['predicted_sentiment'].value_counts().unstack()

#   # Handle potential missing months (if data is sparse)
#   monthly_counts = monthly_counts.fillna(0)  # Fill missing months with 0 count

#   # Create a line chart
#   monthly_counts.plot(kind='line', figsize=(10, 6))
#   plt.xlabel('Month')
#   plt.ylabel('Count')
#   plt.title(f'Sentiment Trend ({year})')
#   plt.legend(title='Sentiment')
#   plt.show()

#   # Create image data for potential display (optional)
#   buf = BytesIO()
#   plt.savefig(buf, format='png')
#   plt.close()
#   data = base64.b64encode(buf.getvalue()).decode('utf-8')
#   return data  # You can return the image data for further use (e.g., display in a layout)

# # Connect dropdown selection to plot update
# year_dropdown.observe(update_plot, names="value")

# # Display the initial plot
# update_plot(year_dropdown.value)

# # Display the dropdown (you might need to adjust layout in your notebook)
# widgets.interact(update_plot, year=year_dropdown)


from django.http import JsonResponse

# @app.route('/update_plot', methods=['POST'])
# def update_plot_post():
#     year = int(request.POST.get('year'))
#     # ... (rest of your update_plot function)
#     plot_data = update_plot(year)
#     return JsonResponse({'plot_data': plot_data})

import pandas as pd
import matplotlib.pyplot as plt


# def plot_negative_reviews_pareto(df):
#     sentiment_counts, merged_df = departments(df)

#     # merged_df.to_csv('merged_reviews.csv', index=False)
    
#     # Filter for negative sentiment reviews
#     negative_reviews_df = merged_df[merged_df['predicted_sentiment'] == 'NEGATIVE']

#     # Count the number of negative reviews for each department
#     department_counts = {
#         'product_quality': negative_reviews_df['product_quality'].sum(),
#         'customer_service':negative_reviews_df['customer_service'].sum(),
#         'purchase_experience':negative_reviews_df['purchase_experience'].sum(),
#         'pricing': negative_reviews_df['pricing'].sum(),
#         'delivery': negative_reviews_df['delivery'].sum(),
#     }
    
#     # Create a DataFrame from the counts
#     department_counts_df = pd.DataFrame(department_counts.items(), columns=['Department', 'Negative_Reviews'])

#     # Sort the DataFrame by Negative Reviews in descending order
#     department_counts_df = department_counts_df.sort_values(by='Negative_Reviews', ascending=False)

#     # Calculate cumulative percentage
#     department_counts_df['Cumulative'] = department_counts_df['Negative_Reviews'].cumsum()
#     department_counts_df['Cumulative Percentage'] = department_counts_df['Cumulative'] / department_counts_df['Negative_Reviews'].sum() * 100

#     # Determine the threshold for highlighting (80%)
#     threshold = 80
#     highlight = department_counts_df[department_counts_df['Cumulative Percentage'] <= threshold]

#     # Create the Pareto chart
#     # fig, ax1 = plt.subplots()
#     fig, ax1 = plt.subplots(figsize=(10,6))

#     # Bar colors based on cumulative percentage
#     colors = ['lightcoral' if dep in highlight['Department'].values else 'lightgray' for dep in department_counts_df['Department']]

#     # Bar chart for negative reviews
#     ax1.bar(department_counts_df['Department'], department_counts_df['Negative_Reviews'], color=colors, alpha=0.8, label='Negative Reviews')
#     ax1.set_ylabel('Number of Reviews', color='b',fontsize=12)
#     ax1.tick_params(axis='y', labelcolor='b')

#     plt.xticks(rotation=30,fontsize=12)

#     # Create a second y-axis for cumulative percentage
#     ax2 = ax1.twinx()
#     ax2.plot(department_counts_df['Department'], department_counts_df['Cumulative Percentage'], color='r', marker='o', label='Cumulative %')
#     ax2.set_ylabel('Cumulative Percentage (%)', color='r',fontsize=12)
#     ax2.tick_params(axis='y', labelcolor='r')
#     ax2.axhline(80, color='gray', linestyle='--')  # 80% line

#     # Adding titles and legends
#     plt.title('Number of Negative Reviews By Department',fontsize=14)
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data





    # return filtered_df

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main Classify review function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# def classify_reviews(df,years):

#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])

#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

#     # Define the keyword lists
#     quality_keywords = ["quality", 'wilt', 'water', 'bulb', 'droopy', 'bruise', 'damage', 'bloom', 'long lasting', 'love', 'perfect', 'look as', 'old', 'dead', 'fresh', 'fabulous']
#     delivery_keywords = ['deliver', 'shipping', 'arrive', 'arrived', 'on time', 'late', 'DOA', 'dead on arrival', 'shipment', 'box', 'fedex', 'fd']
#     subscription_service = ['subscription']
#     purchase_experience = ['easy', 'excellent', 'disappoint', 'stunning', 'Thank you', 'Arrang', 'checkout', 'purchase', 'buy', 'order', 'transaction', 'awesome', 'gorgeous', 'experience', 'selection', 'arrangement', 'packag', 'card', 'beautiful', 'amazing', 'great', 'great flower', 'impres', 'recommend', 'exceptional', 'please', 'gift', 'refund', 'store credit']
#     web_site = ['website', 'promo']
#     customer_service = ['support', 'help', 'service', 'staff', 'representative', 'communication', 'customer']
#     product_pricing = ['competitive', 'price', 'cost', 'expensive', 'cheap', 'value', 'money', 'affordable', 'overpri', r'\$']
#     marketing = ['advertise', 'email']

#     # Compile regex patterns for each keyword list
#     def compile_patterns(keywords):
#         return re.compile(r'\b(' + '|'.join(keywords) + r')\w*\b', re.IGNORECASE)

  
#     product_pricing_pattern = compile_patterns(product_pricing)
#     quality_pattern = compile_patterns(quality_keywords)
#     delivery_pattern = compile_patterns(delivery_keywords)
#     subscription_pattern = compile_patterns(subscription_service)
#     web_site_pattern = compile_patterns(web_site)
#     customer_service_pattern = compile_patterns(customer_service)
#     marketing_pattern = compile_patterns(marketing)
#     purchase_experience_pattern = compile_patterns(purchase_experience)


#     # Initialize the 'department' column with None or empty string
#     filtered_df['department'] = "Other"

#     # Loop through the reviews and update the department column
#     for index, row in filtered_df.iterrows():
#         if pd.notna(row['review']) and isinstance(row['review'], str):
        
#             review = row['review']
            
        
#             if quality_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Product Quality"
#             elif delivery_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Delivery Service"
#             elif subscription_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Subscription Service"
#             elif purchase_experience_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Purchase Experience"
#             elif web_site_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "WebSite Experience"
#             elif customer_service_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Customer Service"
#             elif product_pricing_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Product Pricing"
#             elif marketing_pattern.search(review):
#                 filtered_df.at[index, 'department'] = "Marketing"
#             else:
#                 filtered_df.at[index, 'department'] = "Other"


#     count_positive_product_quality = 0
#     count_positive_delivery_service = 0
#     count_positive_subscription_experience = 0
#     count_positive_purchase_experience = 0
#     count_positive_website_experience = 0
#     count_positive_customer_service= 0
#     count_positive_product_pricing= 0
#     count_positive_marketing= 0

#     count_negative_product_quality = 0
#     count_negative_delivery_service = 0
#     count_negative_subscription_experience = 0
#     count_negative_purchase_experience = 0
#     count_negative_website_experience = 0
#     count_negative_customer_service= 0
#     count_negative_product_pricing= 0
#     count_negative_marketing= 0

#     positive_counts = {
#     "Product Quality": 0,
#     "Delivery Service": 0,
#     "Subscription Service": 0,
#     "Purchase Experience": 0,
#     "WebSite Experience": 0,
#     "Customer Service": 0,
#     "Product Pricing": 0,
#     "Marketing": 0
#     }

#     negative_counts = {
#         "Product Quality": 0,
#         "Delivery Service": 0,
#         "Subscription Service": 0,
#         "Purchase Experience": 0,
#         "WebSite Experience": 0,
#         "Customer Service": 0,
#         "Product Pricing": 0,
#         "Marketing": 0
#     }


#     for index, row in filtered_df.iterrows():
#         department = row["department"]
        
#         if row["predicted_sentiment"] == "POSITIVE":
#             if department in positive_counts:
#                 positive_counts[department] += 1
                
#         elif row["predicted_sentiment"] == "NEGATIVE":
#             if department in negative_counts:
#                 negative_counts[department] += 1

#     # Extract counts for easier access
#     count_positive_product_quality = positive_counts["Product Quality"]
#     count_positive_delivery_service = positive_counts["Delivery Service"]
#     count_positive_subscription_experience = positive_counts["Subscription Service"]
#     count_positive_purchase_experience = positive_counts["Purchase Experience"]
#     count_positive_website_experience = positive_counts["WebSite Experience"]
#     count_positive_customer_service = positive_counts["Customer Service"]
#     count_positive_product_pricing = positive_counts["Product Pricing"]
#     count_positive_marketing = positive_counts["Marketing"]

#     count_negative_product_quality = negative_counts["Product Quality"]
#     count_negative_delivery_service = negative_counts["Delivery Service"]
#     count_negative_subscription_experience = negative_counts["Subscription Service"]
#     count_negative_purchase_experience = negative_counts["Purchase Experience"]
#     count_negative_website_experience = negative_counts["WebSite Experience"]
#     count_negative_customer_service = negative_counts["Customer Service"]
#     count_negative_product_pricing = negative_counts["Product Pricing"]
#     count_negative_marketing = negative_counts["Marketing"]


#     return {
#         "Positive_WebSite_Experience":count_positive_website_experience,
#         "Positive_Subscription_Service":count_positive_subscription_experience,
#         "positive_product_quality": count_positive_product_quality,
#         "positive_customer_service": count_positive_customer_service,
#         "positive_purchase_experience": count_positive_purchase_experience,
#         "positive_pricing": count_positive_product_pricing,
#         "positive_delivery": count_positive_delivery_service,
#         "positive_Marketing":count_positive_marketing,



#         "negative_product_quality": count_negative_product_quality,
#         "negative_customer_service": count_negative_customer_service,
#         "negative_purchase_experience": count_negative_purchase_experience,
#         "negative_pricing": count_negative_product_pricing,
#         "negative_delivery": count_negative_delivery_service,
#         "Negative_Subscription_Service":count_negative_subscription_experience,
#         "Negative_WebSite_Experience":count_negative_website_experience,
#         "Negative_Marketing":count_negative_marketing,



#     }, filtered_df

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<< Dynamic department name from config file  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def classify_reviews(df, years, selected_industry, config_path=r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"):
    # Load the keyword configuration based on the selected industry
    with open(config_path, "r") as f:
        config = json.load(f)

    industry_config = config["industries"][selected_industry]["departments"]

    # Compile regex patterns for each department
    def compile_patterns(keywords):
        return re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, keywords))), re.IGNORECASE)

    patterns = {dept: compile_patterns(keywords) for dept, keywords in industry_config.items()}

    # Filter DataFrame by the specified years
    df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

    # Initialize the 'department' column
    filtered_df['department'] = "Other"

    # Classify reviews by department based on keyword matching
    for index, row in filtered_df.iterrows():
        if pd.notna(row['review']) and isinstance(row['review'], str):
            review = row['review']
            for dept, pattern in patterns.items():
                if pattern.search(review):
                    filtered_df.at[index, 'department'] = dept
                    break

    # Initialize counts for positive and negative reviews per department
    positive_counts = {dept: 0 for dept in industry_config.keys()}
    negative_counts = {dept: 0 for dept in industry_config.keys()}

    # Count positive and negative reviews for each department
    for index, row in filtered_df.iterrows():
        department = row["department"]
        
        if row["predicted_sentiment"] == "POSITIVE" and department in positive_counts:
            positive_counts[department] += 1
        elif row["predicted_sentiment"] == "NEGATIVE" and department in negative_counts:
            negative_counts[department] += 1

    # Prepare the return format with dynamic department names
    result = {}

    for dept in industry_config.keys():
        # Generate dynamic keys for positive and negative counts
        positive_key = f"positive_{dept.replace(' ', '_').lower()}"
        negative_key = f"negative_{dept.replace(' ', '_').lower()}"
        
        result[positive_key] = positive_counts.get(dept, 0)
        result[negative_key] = negative_counts.get(dept, 0)

    filtered_df.to_csv(r"industriesbasefilenewwww.csv", index=False)


    return result, filtered_df

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def compile_patterns(keywords):
    return re.compile(r'\b(' + '|'.join(keywords) + r')\w*\b', re.IGNORECASE)

import json


# def classify_reviews(df, years, industry, config_path=r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"):
#     # Load the keyword configuration based on the selected industry
#     with open(fr"{config_path}", "r") as f: 
#         config = json.load(f)
    
#     industry_config = config["industries"][selected_industry]["departments"]

#     # Compile regex patterns for each department
#     patterns = {dept: compile_patterns(keywords) for dept, keywords in industry_config.items()}

#     # Filter DataFrame by the specified years
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

#     # Initialize the 'department' column
#     filtered_df['department'] = "Other"

#     # Classify reviews by department based on keyword matching
#     for index, row in filtered_df.iterrows():
#         if pd.notna(row['review']) and isinstance(row['review'], str):
#             review = row['review']
#             for dept, pattern in patterns.items():
#                 if pattern.search(review):
#                     filtered_df.at[index, 'department'] = dept
#                     break

#     # Initialize counts for positive and negative reviews per department
#     positive_counts = {dept: 0 for dept in industry_config.keys()}
#     negative_counts = {dept: 0 for dept in industry_config.keys()}

#     # Count positive and negative reviews for each department
#     for index, row in filtered_df.iterrows():
#         department = row["department"]
        
#         if row["predicted_sentiment"] == "POSITIVE" and department in positive_counts:
#             positive_counts[department] += 1
#         elif row["predicted_sentiment"] == "NEGATIVE" and department in negative_counts:
#             negative_counts[department] += 1

#     # Prepare the return format
#     return {
#     # Positive counts
#     "positive_product_quality": positive_counts.get("Product Quality", 0),
#     "positive_delivery": positive_counts.get("Delivery Service", 0),
#     "Positive_Subscription_Service": positive_counts.get("Subscription Service", 0),
#     "positive_purchase_experience": positive_counts.get("Purchase Experience", 0),
#     "Positive_WebSite_Experience": positive_counts.get("WebSite Experience", 0),
#     "positive_customer_service": positive_counts.get("Customer Service", 0),
#     "positive_pricing": positive_counts.get("Product Pricing", 0),
#     "positive_Marketing": positive_counts.get("Marketing", 0),

#     # Negative counts
#     "negative_product_quality": negative_counts.get("Product Quality", 0),
#     "negative_delivery": negative_counts.get("Delivery Service", 0),
#     "Negative_Subscription_Service": negative_counts.get("Subscription Service", 0),
#     "negative_purchase_experience": negative_counts.get("Purchase Experience", 0),
#     "Negative_WebSite_Experience": negative_counts.get("WebSite Experience", 0),
#     "negative_customer_service": negative_counts.get("Customer Service", 0),
#     "negative_pricing": negative_counts.get("Product Pricing", 0),
#     "Negative_Marketing": negative_counts.get("Marketing", 0),
# }, filtered_df







def wrap_text(text, line_length=50):
    words = text.split()
    wrapped_text = ""
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            wrapped_text += current_line + "\n"
            current_line = word
        else:
            current_line += (word + " ")
    
    wrapped_text += current_line  # Add the last line
    return wrapped_text.strip()




def plot_negative_reviews_pareto(df, years,selected_industry):
    # Classify reviews and get the merged DataFrame
    # sentiment_counts, merged_df = classify_reviews(df, years,selected_industry)
    sentiment_counts, merged_df = classify_reviews_BarChartPN_plot(df, years, selected_industry)

    # Convert RESPONSE_TIMESTAMP to datetime
    merged_df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])

    print("Selected years:", years)

    # Filter data for the selected years
    filtered_df = merged_df[merged_df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]
    filtered_df = filtered_df[filtered_df['department'] != 'Other']

    # Filter for negative sentiment reviews
    negative_reviews_df = filtered_df[filtered_df['predicted_sentiment'] == 'NEGATIVE']

    print(negative_reviews_df.head(10))

    # Count negative reviews by department
    # department_counts = negative_reviews_df.groupby('department')['SCORE'].sum().reset_index()
    department_counts = negative_reviews_df['department'].value_counts().reset_index()

    print("department_counts",department_counts)

    # Rename columns for clarity
    department_counts.columns = ['Department', 'Negative_Reviews']

    # Sort the DataFrame by Negative Reviews in descending order
    department_counts = department_counts.sort_values(by='Negative_Reviews', ascending=False)

    # Calculate cumulative percentage
    department_counts['Cumulative'] = department_counts['Negative_Reviews'].cumsum()
    department_counts['Cumulative Percentage'] = department_counts['Cumulative'] / department_counts['Negative_Reviews'].sum() * 100

    # Determine the threshold for highlighting (80%)
    threshold = 80
    highlight = department_counts[department_counts['Cumulative Percentage'] <= threshold]

    # Create the Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # Bar colors based on cumulative percentage
    colors = ['lightcoral' if dep in highlight['Department'].values else 'lightgray' for dep in department_counts['Department']]

    # Bar chart for negative reviews
    ax1.bar(department_counts['Department'], department_counts['Negative_Reviews'], color=colors, alpha=0.8, label='Negative Reviews')
    ax1.set_ylabel('Number of Reviews', color='b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Apply custom labels
    ax1.set_xticklabels(department_counts['Department'], rotation=45, ha='right',fontweight='bold')

    # Create a second y-axis for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(department_counts['Department'], department_counts['Cumulative Percentage'], color='r', marker='o', label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage (%)', color='r', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(80, color='gray', linestyle='--')  # 80% line

    # Adding titles and legends
    # plt.title(f'Number of Negative Reviews By Department {years}', fontsize=14, fontweight='bold')
    wrapped_title = wrap_text(f'Number of Negative Reviews By Department {years}', line_length=50)
    plt.title(wrapped_title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return data



# def plot_negative_reviews_pareto(df, years):
#     # Classify reviews and get the merged DataFrame
#     sentiment_counts, merged_df = classify_reviews(df, years)

#     # Convert RESPONSE_TIMESTAMP to datetime
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])

#     print("Selected years:", years)

#     # Filter data for the selected years
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

#     # Filter for negative sentiment reviews
#     negative_reviews_df = filtered_df[filtered_df['predicted_sentiment'] == 'NEGATIVE']

#     print(negative_reviews_df.head(10))

#     # Count negative reviews by department
#     # department_counts = negative_reviews_df.groupby('department')['SCORE'].sum().reset_index()
#     department_counts = negative_reviews_df['department'].value_counts().reset_index()

#     # Rename columns for clarity
#     department_counts.columns = ['Department', 'Negative_Reviews']

#     # Sort the DataFrame by Negative Reviews in descending order
#     department_counts = department_counts.sort_values(by='Negative_Reviews', ascending=False)

#     # Calculate cumulative percentage
#     department_counts['Cumulative'] = department_counts['Negative_Reviews'].cumsum()
#     department_counts['Cumulative Percentage'] = department_counts['Cumulative'] / department_counts['Negative_Reviews'].sum() * 100

#     # Determine the threshold for highlighting (80%)
#     threshold = 80
#     highlight = department_counts[department_counts['Cumulative Percentage'] <= threshold]

#     # Create the Pareto chart
#     fig, ax1 = plt.subplots(figsize=(10, 8))

#     # Bar colors based on cumulative percentage
#     colors = ['lightcoral' if dep in highlight['Department'].values else 'lightgray' for dep in department_counts['Department']]

#     # Bar chart for negative reviews
#     ax1.bar(department_counts['Department'], department_counts['Negative_Reviews'], color=colors, alpha=0.8, label='Negative Reviews')
#     ax1.set_ylabel('Number of Reviews', color='b', fontsize=12, fontweight='bold')
#     ax1.tick_params(axis='y', labelcolor='b')

#     plt.xticks(rotation=45, fontsize=12)

#     # Apply custom labels
#     ax1.set_xticklabels(department_counts['Department'], rotation=45, ha='right')

#     # Create a second y-axis for cumulative percentage
#     ax2 = ax1.twinx()
#     ax2.plot(department_counts['Department'], department_counts['Cumulative Percentage'], color='r', marker='o', label='Cumulative %')
#     ax2.set_ylabel('Cumulative Percentage (%)', color='r', fontsize=12, fontweight='bold')
#     ax2.tick_params(axis='y', labelcolor='r')
#     ax2.axhline(80, color='gray', linestyle='--')  # 80% line

#     # Adding titles and legends
#     plt.title('Number of Negative Reviews By Department', fontsize=14, fontweight='bold')
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()

#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     return data







import textwrap



# <<<<<<<<<<<<<<<<<<<<  for clustering base code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# def plot_negative_reviews_pareto():
#     # sentiment_counts, merged_df = departments(df)

#     merged_df = pd.read_csv(r'C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\customer_segments.csv')

#     # Filter for negative sentiment reviews
#     negative_reviews_df = merged_df[merged_df['predicted_sentiment'] == 'NEGATIVE']

#     # Count the number of negative reviews for each department from the Customer_Segment column
#     department_counts = negative_reviews_df['Customer_Segment'].value_counts()

#     # Create a DataFrame from the counts
#     department_counts_df = department_counts.reset_index()
#     department_counts_df.columns = ['Department', 'Negative_Reviews']

#     # Sort the DataFrame by Negative Reviews in descending order
#     department_counts_df = department_counts_df.sort_values(by='Negative_Reviews', ascending=False)

#     # Calculate cumulative percentage
#     department_counts_df['Cumulative'] = department_counts_df['Negative_Reviews'].cumsum()
#     department_counts_df['Cumulative Percentage'] = department_counts_df['Cumulative'] / department_counts_df['Negative_Reviews'].sum() * 100

#     # Determine the threshold for highlighting (80%)
#     threshold = 80
#     highlight = department_counts_df[department_counts_df['Cumulative Percentage'] <= threshold]

#     # Create the Pareto chart
#     fig, ax1 = plt.subplots(figsize=(10, 8))

#     # Bar colors based on cumulative percentage
#     colors = ['lightcoral' if dep in highlight['Department'].values else 'lightgray' for dep in department_counts_df['Department']]

#     # Bar chart for negative reviews
#     ax1.bar(department_counts_df['Department'], department_counts_df['Negative_Reviews'], color=colors, alpha=0.8, label='Negative Reviews')
#     ax1.set_ylabel('Number of Reviews', color='b', fontsize=12, fontweight='bold')
#     ax1.tick_params(axis='y', labelcolor='b')

#     plt.xticks(rotation=0, fontsize=12)

#     # Custom labels mapping
#     custom_labels_mapping = {
#         'Customer Service': 'Customer Service',
#         'Delivery': 'Delivery',
#         'Pricing': 'Pricing',
#         'Product Quality': 'Product Quality',
#         'Purchase Experience': 'Purchase Experience'
#     }

#     # Apply custom labels based on the mapping
#     custom_labels = [textwrap.fill(custom_labels_mapping[dep], width=15) for dep in department_counts_df['Department']]

#     ax1.set_xticklabels(custom_labels, rotation=0, fontsize=12)

#     # Create a second y-axis for cumulative percentage
#     ax2 = ax1.twinx()
#     ax2.plot(department_counts_df['Department'], department_counts_df['Cumulative Percentage'], color='r', marker='o', label='Cumulative %')
#     ax2.set_ylabel('Cumulative Percentage (%)', color='r', fontsize=12, fontweight='bold')
#     ax2.tick_params(axis='y', labelcolor='r')
#     ax2.axhline(80, color='gray', linestyle='--')  # 80% line

#     # Adding titles and legends
#     plt.title('Number of Negative Reviews By Department', fontsize=14, fontweight='bold')
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()
    
    # Save the plot to a buffer
    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # plt.close()
    # data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # return data

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Example usage
# df = pd.DataFrame({
#     'Department': ['A', 'A', 'B', 'C', 'B', 'A'],
#     'review': ['bad', 'terrible', 'not good', 'poor', 'bad', 'awful'],
#     'predicted_sentiment': ['negative', 'negative', 'negative', 'negative', 'negative', 'negative']
# })
# plot_negative_reviews_pareto(df)


# <<<<<<<<<<<<<<<<<<<<<<  n - gram   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def generate_bigram_analysis(df):

    sentiment_counts, merged_df = classify_reviews(df, [2016, 2017, 2018, 2019])
    """
    Generate top bigrams for negative reviews by department and calculate average review length.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing columns 'department', 'review', and 'predicted_sentiment'.
    
    Returns:
        dict: A dictionary with each department as a key, containing:
              - 'top_bigrams': DataFrame of the top bigrams with counts
              - 'average_length': Average review length for negative reviews
    """
    # List of unique departments
    merged_df = merged_df[merged_df['department'] != 'Other']
    departments = merged_df['department'].unique()
    
    # Create a dictionary to hold results for each department
    bigram_results = {}
    results_list = []
    # Process each department
    for department in departments:
        # Step 1: Filter for negative reviews in the current department
        negative_reviews = merged_df[(merged_df['department'] == department) & (merged_df['predicted_sentiment'] == 'NEGATIVE')]['review']
        
        # Check if there are any negative reviews
        if len(negative_reviews) > 0:
            # Step 2: Create bigrams with stopword removal
            vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')  # Using default English stopwords
            bigrams = vectorizer.fit_transform(negative_reviews)
            bigram_counts = bigrams.sum(axis=0)

            # Step 3: Create a DataFrame for bigrams and sort by count
            bigram_df = pd.DataFrame(bigram_counts.T, index=vectorizer.get_feature_names_out(), columns=['Count'])
            bigram_df = bigram_df.sort_values(by='Count', ascending=False)

            # Get top 10 bigrams
            top_bigrams = bigram_df.head(10)

            print("dfdfdfdf",top_bigrams)

        #     top_bigrams_list = [f"{index:<40} {count}" for index, count in zip(top_bigrams.index, top_bigrams['Count'])]

        #     # Add the formatted bigram results to the final list
        #     results_list.extend(top_bigrams_list)
        # else:
        #     # If no negative reviews are found, add a message indicating so
        #     results_list.append(f"No negative reviews found for {department}.")

            # Optional: Calculate the average review length for negative reviews in the department
            merged_df['Review Length'] = merged_df['review'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)

            average_length = merged_df[(merged_df['department'] == department) & (merged_df['predicted_sentiment'] == 'NEGATIVE')]['Review Length'].mean()

            # Store results in the dictionary
            # bigram_results[department] = {
            #     'top_bigrams': top_bigrams,
            #     # 'average_length': average_length
            # }
            bigram_results[department] = top_bigrams['Count'].to_dict()
        else:
            # If no negative reviews are found, store a message
            # bigram_results[department] = {
            #     'top_bigrams': pd.DataFrame(),  # Empty DataFrame for bigrams
            #     'average_length': None
            # }
            bigram_results[department] = {}
            
    return bigram_results
    


# def generate_html_table(bigram_results):
#     """
#     Generates an HTML table from the bigram analysis results.
    
#     Parameters:
#         bigram_results (dict): The dictionary returned from `generate_bigram_analysis`.
    
#     Returns:
#         str: HTML string of the table.
#     """
#     html = '''
#     <table style="width:100%; border-collapse: collapse;">
#         <tr>
#             <th colspan="3" style="border: 1px solid black; padding: 8px; text-align: center; font-weight: bold;">
#                 N-gram Analysis (3:3) of Negative Review By Department
#             </th>
#         </tr>
#     '''

#     # Split departments into rows of three for a structured layout
#     departments = list(bigram_results.keys())
#     for i in range(0, len(departments), 3):
#         row_departments = departments[i:i+3]
        
#         # Header row
#         html += '<tr>'
#         for department in row_departments:
#             html += f'<th style="border: 1px solid black; padding: 8px; text-align: center;">Top 10 Negative Reviews of {department}</th>'
#         html += '</tr>'
        
#         # Data rows
#         for j in range(10):
#             html += '<tr>'
#             for department in row_departments:
#                 if not bigram_results[department]['top_bigrams'].empty:
#                     # Get the bigram and count if it exists, otherwise empty cells
#                     if j < len(bigram_results[department]['top_bigrams']):
#                         bigram = bigram_results[department]['top_bigrams'].index[j]
#                         count = bigram_results[department]['top_bigrams'].iloc[j]['Count']
#                         html += f'<td style="border: 1px solid black; padding: 8px; text-align: center;">{bigram} ({count})</td>'
#                     else:
#                         html += '<td style="border: 1px solid black; padding: 8px; text-align: center;"></td>'
#                 else:
#                     html += '<td style="border: 1px solid black; padding: 8px; text-align: center;">No Data</td>'
#             html += '</tr>'

#     html += '</table>'
#     return html

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer

import os
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer

# def plot_bigrams_by_sentiment(df, years):
#     # Ensure the figures directory exists
#     if not os.path.exists('figures'):
#         os.makedirs('figures')

#     # Filter out 'Other' department
#     sentiment_counts, merged_df = classify_reviews(df, [2016, 2017, 2018, 2019])
#     merged_df = merged_df[merged_df['department'] != 'Other']
#     departments = merged_df['department'].unique()

#     # Create bigram results structure
#     bigram_results = {dept: {'positive': {}, 'negative': {}} for dept in departments}

#     # Process each department for both positive and negative reviews
#     for department in departments:
#         for sentiment in ['POSITIVE', 'NEGATIVE']:
#             reviews = merged_df[(merged_df['department'] == department) & (merged_df['predicted_sentiment'] == sentiment)]['review']
            
#             if len(reviews) > 0:
#                 # Create bigrams
#                 vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
#                 bigrams = vectorizer.fit_transform(reviews)
#                 bigram_counts = bigrams.sum(axis=0)

#                 # Create DataFrame for bigrams and sort by count
#                 bigram_df = pd.DataFrame(bigram_counts.T, index=vectorizer.get_feature_names_out(), columns=['Count'])
#                 top_bigrams = bigram_df.sort_values(by='Count', ascending=False).head(10)
#                 bigram_results[department][sentiment.lower()] = top_bigrams['Count'].to_dict()  # Store the top bigrams

#     # Function to create and save plots for a subset of departments
#     def create_and_save_plot(departments_subset, fig_title, fig_num):
#         num_departments = len(departments_subset)
#         num_plots = num_departments * 2  # Two plots per department (positive and negative)
#         cols = 2  # Two columns (one for positive, one for negative)
#         rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

#         fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 16))
#         axes = axes.flatten()
        
#         for i, department in enumerate(departments_subset):
#             for j, sentiment in enumerate(['positive', 'negative']):
#                 bigram_counts = bigram_results[department][sentiment]
#                 bigrams = list(bigram_counts.keys())
#                 counts = list(bigram_counts.values())

#                 # Set color based on sentiment
#                 colors = plt.cm.Blues(np.linspace(0.3, 1, len(bigrams))) if sentiment == 'positive' else plt.cm.Reds(np.linspace(0.3, 1, len(bigrams)))

#                 # Calculate subplot index and plot
#                 ax_idx = i * 2 + j
#                 if ax_idx < len(axes):
#                     ax = axes[ax_idx]
#                     if bigrams:  # Only plot if there are bigrams found
#                         ax.barh(bigrams, counts, color=colors)
#                         ax.set_title(f'Top {sentiment.capitalize()} Trigram in {department}', fontsize=8)
#                         ax.invert_yaxis()  # Highest count on top
#                         ax.set_yticklabels(bigrams, fontsize=8)  # Adjust font size for y-tick labels
#                         ax.tick_params(axis='x', labelsize=8)  # Set x-tick labels font size

#         # Hide any unused subplots if the grid is larger than needed
#         for k in range(ax_idx + 1, len(axes)):
#             axes[k].axis('off')

#         # Add spacing between subplots and set the main title
#         fig.suptitle(fig_title, fontsize=14)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.subplots_adjust(hspace=0.25, wspace=0.5)

#         # Save the figure
#         fig.savefig(f'figures/trigram_analysis_part_{fig_num}.png')
#         plt.close(fig)  # Close the figure to avoid display in Jupyter

#     # Create and save figures for each half of the departments
#     num_parts = (len(departments) + 1) // 2  # Calculate number of parts needed
#     print("num_parts",num_parts)
#     for i in range(num_parts):
#         create_and_save_plot(departments[i*2:(i+1)*2], f"Trigram Analysis - Part {i+1}", i + 1)

#         print("welcome to mon thhhhhhhhhhhhh")

#     return [f'figures/trigram_analysis_part_{i+1}.png' for i in range(num_parts)]  # Return list of figure paths


# def plot_bigrams_by_sentiment(df,years):
#     # Filter out 'Other' department
#     sentiment_counts, merged_df = classify_reviews(df, [2016, 2017, 2018, 2019])
#     merged_df = merged_df[merged_df['department'] != 'Other']
#     departments = merged_df['department'].unique()

#     # Divide the departments into halves (adjust according to your number of departments)
#     first_half = departments[:2]
#     second_half = departments[2:4]
#     third_half = departments[4:6]
#     fourth_half = departments[6:8]

#     bigram_results = {dept: {'positive': {}, 'negative': {}} for dept in departments}

#     # Process each department for both positive and negative reviews
#     for department in departments:
#         for sentiment in ['POSITIVE', 'NEGATIVE']:
#             reviews = merged_df[(merged_df['department'] == department) & (merged_df['predicted_sentiment'] == sentiment)]['review']
            
#             if len(reviews) > 0:
#                 # Create bigrams
#                 vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
#                 bigrams = vectorizer.fit_transform(reviews)
#                 bigram_counts = bigrams.sum(axis=0)

#                 # Create DataFrame for bigrams and sort by count
#                 bigram_df = pd.DataFrame(bigram_counts.T, index=vectorizer.get_feature_names_out(), columns=['Count'])
#                 top_bigrams = bigram_df.sort_values(by='Count', ascending=False).head(10)
#                 bigram_results[department][sentiment.lower()] = top_bigrams['Count'].to_dict()  # Store the top bigrams

#     # Function to create plots for a subset of departments
#     def create_plot(departments_subset, fig_title):
#         num_departments = len(departments_subset)
#         num_plots = num_departments * 2  # Two plots per department (positive and negative)
#         cols = 2  # Two columns (one for positive, one for negative)
#         rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

#         fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 16))
#         axes = axes.flatten()
        
#         for i, department in enumerate(departments_subset):
#             for j, sentiment in enumerate(['positive', 'negative']):
#                 bigram_counts = bigram_results[department][sentiment]
#                 bigrams = list(bigram_counts.keys())
#                 counts = list(bigram_counts.values())

#                 # Set color based on sentiment
#                 colors = plt.cm.Blues(np.linspace(0.3, 1, len(bigrams))) if sentiment == 'positive' else plt.cm.Reds(np.linspace(0.3, 1, len(bigrams)))

#                 # Calculate subplot index and plot
#                 ax_idx = i * 2 + j
#                 if ax_idx < len(axes):
#                     ax = axes[ax_idx]
#                     if bigrams:  # Only plot if there are bigrams found
#                         ax.barh(bigrams, counts, color=colors)
#                         ax.set_title(f'Top {sentiment.capitalize()} Trigram in {department}', fontsize=8)
#                         ax.invert_yaxis()  # Highest count on top
#                         ax.set_yticklabels(bigrams, fontsize=8)  # Adjust font size for y-tick labels
#                         ax.tick_params(axis='x', labelsize=8)  # Set x-tick labels font size

#         # Hide any unused subplots if the grid is larger than needed
#         for k in range(ax_idx + 1, len(axes)):
#             axes[k].axis('off')

#         # Add spacing between subplots and set the main title
#         fig.suptitle(fig_title, fontsize=14)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.subplots_adjust(hspace=0.25, wspace=0.5)
        
#         return fig  # Return the figure object

#     # Create figures and return them
#     figures = []
#     figures.append(create_plot(first_half, "Trigram Analysis - First Half"))
#     figures.append(create_plot(second_half, "Trigram Analysis - Second Half"))
#     figures.append(create_plot(third_half, "Trigram Analysis - Third Half"))
#     figures.append(create_plot(fourth_half, "Trigram Analysis - Fourth Half"))
    
#     return figures  # Return the list of figures

counting=1
# def plot_bigrams_by_sentiment(df, years,selected_industry):
#     # Ensure the figures directory exists (optional, as we're not saving to disk here)
#     if not os.path.exists('figures'):
#         os.makedirs('figures')

# #     # Filter out 'Other' department
# #     # sentiment_counts, merged_df = classify_reviews(df, years,selected_industry)
#     sentiment_counts, merged_df = classify_reviews_BarChartPN_plot(df, years, selected_industry)
#     merged_df = merged_df[merged_df['department'] != 'Other']
#     merged_df['RESPONSE_TIMESTAMP'] = pd.to_datetime(merged_df['RESPONSE_TIMESTAMP'])
#     filtered_df = merged_df[merged_df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]
#     departments = filtered_df['department'].unique()

# #     # Create bigram results structure
#     bigram_results = {dept: {'positive': {}, 'negative': {}} for dept in departments}

# #     # Process each department for both positive and negative reviews
#     for department in departments:
#         for sentiment in ['POSITIVE', 'NEGATIVE']:
#             reviews = filtered_df[(filtered_df['department'] == department) & (filtered_df['predicted_sentiment'] == sentiment)]['review']
            
#             if len(reviews) > 0:
#                 # Create bigrams
#                 vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
#                 bigrams = vectorizer.fit_transform(reviews)
#                 bigram_counts = bigrams.sum(axis=0)

#                 # Create DataFrame for bigrams and sort by count
#                 bigram_df = pd.DataFrame(bigram_counts.T, index=vectorizer.get_feature_names_out(), columns=['Count'])
#                 top_bigrams = bigram_df.sort_values(by='Count', ascending=False).head(10)
#                 bigram_results[department][sentiment.lower()] = top_bigrams['Count'].to_dict()  # Store the top bigrams

    # Function to create and save plots for a subset of departments

def filter_bigrams_by_keywords(bigram_index, keywords):
    filtered_index = [
        bigram for bigram in bigram_index 
        if any(word in keywords for word in bigram.split())  # Check word match
    ]
    return filtered_index

def plot_bigrams_by_sentiment(df, years, selected_industry):
    # Ensure the figures directory exists (optional, as we're not saving to disk here)
    if not os.path.exists('figures'):
        os.makedirs('figures')

    sentiment_counts, merged_df = classify_reviews_BarChartPN_plot(df, years, selected_industry)
    merged_df = merged_df[merged_df['department'] != 'Other']
    merged_df['RESPONSE_TIMESTAMP'] = pd.to_datetime(merged_df['RESPONSE_TIMESTAMP'])
    filtered_df = merged_df[merged_df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]
    departments = filtered_df['department'].unique()

    # Create bigram results structure
    bigram_results = {dept: {'positive': {}, 'negative': {}} for dept in departments}

    # Extract matched_keywords
    matched_keywords = set(filtered_df['mainword'].explode().dropna().unique())

    print('matched_keywords',matched_keywords)

    # Process each department for both positive and negative reviews
    for department in departments:
        for sentiment in ['POSITIVE', 'NEGATIVE']:
            reviews = filtered_df[(filtered_df['department'] == department) & 
                                  (filtered_df['predicted_sentiment'] == sentiment)]['review']
            
            if len(reviews) > 0:
                # Create bigrams
                vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')  # Adjust n-gram range if needed
                bigrams = vectorizer.fit_transform(reviews)
                bigram_counts = bigrams.sum(axis=0)

                

                # Create DataFrame for bigrams and sort by count
                bigram_df = pd.DataFrame(bigram_counts.T, index=vectorizer.get_feature_names_out(), columns=['Count'])

                filtered_bigrams = filter_bigrams_by_keywords(bigram_df.index, matched_keywords)

                # Apply the filtering to the new DataFrame
                bigram_df = bigram_df.loc[filtered_bigrams]

                # print('bigram_df',bigram_df.index)
                
                # Filter bigrams by matched_keywords
                # bigram_df = bigram_df[bigram_df.index.isin(matched_keywords)]
                
                # Sort and get top 10 bigrams
                top_bigrams = bigram_df.sort_values(by='Count', ascending=False).head(10)

                # Print or store results
                print(top_bigrams)


                bigram_results[department][sentiment.lower()] = top_bigrams['Count'].to_dict()  # Store the top bigrams


    # (Rest of the function: create_and_save_plot remains unchanged)



    def create_and_save_plot(departments_subset):
        global counting
        num_departments = len(departments_subset)
        num_plots = num_departments * 2  # Two plots per department (positive and negative)
        cols = 2  # Two columns (one for positive, one for negative)
        nrows = (num_plots // 2) + (num_plots % 2 > 0)
        # rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

        fig, axes = plt.subplots(nrows=nrows, ncols=cols, figsize=(16,nrows*7))
        axes = axes.flatten()

        # fig.suptitle("Trigram Analysis by Department", fontsize=16, ha='center',y=0.95)
        # Assume the first subplot is at index 0
          # Get the first subplot axis

        if counting ==1:
            first_ax = axes[0]
        # Add the title only to the first subplot
            first_ax.text(1.5, 1.2, "Trigram Analysis by Department", 
                        fontsize=16, ha='center', va='center', fontweight='bold', transform=first_ax.transAxes)
        # Use plt.figtext to add the title once at the top of all subplots
        # plt.figtext(1.5, 1.05, "Trigram Analysis by Department", 
        #             fontsize=16, ha='center', va='center', fontweight='bold')


        counting+=1
        for i, department in enumerate(departments_subset):
            for j, sentiment in enumerate(['positive', 'negative']):
                bigram_counts = bigram_results[department][sentiment]
                bigrams = list(bigram_counts.keys())
                counts = list(bigram_counts.values())

                # Set color based on sentiment
                colors = plt.cm.Greens(np.linspace(1, 0.3, len(bigrams))) if sentiment == 'positive' else plt.cm.Reds(np.linspace(1, 0.3, len(bigrams)))

                # Calculate subplot index and plot
                ax_idx = i * 2 + j
                if ax_idx < len(axes):
                    ax = axes[ax_idx]
                    if bigrams:  # Only plot if there are bigrams found
                        # Reduce the bar width by setting height < 1.0
                        bar_height = 0.6  # Adjust this value as needed
                        bars = ax.barh(bigrams, counts, color=colors,height=bar_height)
                        ax.invert_yaxis()  # Highest count on top
                        ax.set_yticklabels(bigrams, fontsize=10)  # Adjust font size for y-tick labels
                        ax.tick_params(axis='x', labelsize=10)  # Set x-tick labels font size

                         # Add count at the end of each bar
                        for bar in bars:
                            width = bar.get_width()  # Get the width (count) of the bar
                            ax.text(width, bar.get_y() + bar.get_height() / 2,  # Position the count
                                    f'{width}', va='center', ha='left', fontsize=10)  # Place text with left alignment
                    
                    ax.set_title(f'{sentiment.capitalize()}', fontsize=12,pad=0)
       
                # Add department name above the first column plot for each department
                if j == 0:
                    # ax.set_title(f"{department}", fontsize=14, fontweight='bold', loc='left',pad=20)
                    ax.text(-0.1, 1.1, f"{department}", transform=ax.transAxes,
                        fontsize=14, fontweight='bold', ha='left', va='center')

        # Hide any unused subplots if the grid is larger than needed
        # for k in range(ax_idx + 1, len(axes)):
        #     axes[k].axis('off')

        # Add spacing between subplots and set the main title
        # fig.suptitle(fig_title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.2, wspace=0.5)

        # Save the figure to a BytesIO stream and encode it as base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)  # Close the figure to avoid display in Jupyter
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_image

    # Create and save figures for each half of the departments
    num_parts = (len(departments) + 1) // 2  # Calculate number of parts needed
    base64_images = []
    for i in range(num_parts):
        # base64_image = create_and_save_plot(departments[i*2:(i+1)*2], f"Trigram Analysis - Part {i+1}")
        
        base64_image = create_and_save_plot(departments[i*2:(i+1)*2])
        base64_images.append(base64_image)

    return base64_images


# def plot_yearly_percentage_changes(df, years):
#     # Convert RESPONSE_TIMESTAMP to datetime and extract year and month
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'], utc=True)
#     df['Year'] = df['RESPONSE_TIMESTAMP'].dt.year
#     df['Month'] = df['RESPONSE_TIMESTAMP'].dt.month

#     # Filter for the years of interest
#     df = df[df['Year'].isin(years)]

#     # Calculate month-wise sentiment count
#     monthly_counts = df.groupby(['Year', 'Month', 'predicted_sentiment']).size().unstack(fill_value=0)

#     # Function to calculate percentage changes between two years
#     def calculate_percentage_change(year1, year2):
#         percentage_change_data = {'Month': []}
        
#         for month in range(1, 13):
#             if month in monthly_counts.loc[year1].index and month in monthly_counts.loc[year2].index:
#                 change = (monthly_counts.loc[year2].loc[month] - monthly_counts.loc[year1].loc[month]) / monthly_counts.loc[year1].loc[month] * 100
#                 percentage_change_data['Month'].append(f'{month:02d}-{year1}-{year2}')
#                 for sentiment in change.index:
#                     if sentiment not in percentage_change_data:
#                         percentage_change_data[sentiment] = []
#                     percentage_change_data[sentiment].append(change[sentiment])

#         return pd.DataFrame(percentage_change_data).set_index('Month')

#     # Calculate percentage changes for the given years
#     data_frames = []
#     titles = []
#     for i in range(len(years) - 1):
#         year1, year2 = years[i], years[i + 1]
#         data_frames.append(calculate_percentage_change(year1, year2))
#         titles.append(f'% Change Between Year {year1} & {year2}')

#     # Plot configuration
#     sentiments=['POSITIVE', 'NEUTRAL', 'NEGATIVE']
#     nrows=3
#     ncols=3
#     color_mapping = {'NEGATIVE': 'red', 'NEUTRAL': 'blue', 'POSITIVE': 'green'}
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 14))
#     axes = axes.flatten()

#     # Plot each sentiment for the year combinations
#     for plot_index, (data_frame, title) in enumerate(zip(data_frames, titles)):
#         for i, sentiment in enumerate(sentiments):
#             ax_index = plot_index * len(sentiments) + i
#             if sentiment in data_frame.columns:
#                 axes[ax_index].plot(data_frame.index, data_frame[sentiment], marker='o', 
#                                     label=sentiment, color=color_mapping.get(sentiment, 'blue'))
                
#                 # Annotations for percentage changes
#                 for j in range(len(data_frame)):
#                     # x, y = data_frame.index[j], data_frame[sentiment].iloc[j]
    
#                     # # Get axis limits
#                     # y_min, y_max = axes[ax_index].get_ylim()
#                     # x_min, x_max = axes[ax_index].get_xlim()
                    
#                     # # Adjust offset if annotation is close to boundaries
#                     # y_offset = 5  # Default offset
#                     # if y > y_max - (y_max - y_min) * 0.1:  # If close to top
#                     #     y_offset = -10  # Move annotation below the point
#                     # elif y < y_min + (y_max - y_min) * 0.1:  # If close to bottom
#                     #     y_offset = 10  # Move annotation above the point

#                     # x_offset = 0  # Adjust for x boundaries as well
#                     # if x < x_min + (x_max - x_min) * 0.05:  # Close to left side
#                     #     x_offset = 5
#                     # elif x > x_max - (x_max - x_min) * 0.05:  # Close to right side
#                     #     x_offset = -5

#                     axes[ax_index].annotate(f"{data_frame[sentiment].iloc[j]:.2f}%", 
#                                             (data_frame.index[j], data_frame[sentiment].iloc[j]), 
#                                             textcoords="offset points", 
#                                             xytext=(0, 5), 
#                                             ha='center', 
#                                             fontsize=8, 
#                                             color=axes[ax_index].get_lines()[0].get_color(),
#                                             rotation=45)
                
#                 if sentiment == 'NEUTRAL':
#                     axes[ax_index].set_title(title, fontsize=12 , pad=25)
                
#                 axes[ax_index].set_ylabel('% Change in Sentiment Count', fontsize=12)
#                 axes[ax_index].axhline(0, color='grey', linewidth=0.8, linestyle='--')
#                 axes[ax_index].grid()
#                 axes[ax_index].legend(title='Sentiment', fontsize='small')
#                 axes[ax_index].set_xticks([])

#                 # Add month labels below the plot
#                 for j in range(12):
#                     month_number = j + 1
#                     axes[ax_index].text(j, -30, str(month_number), ha='left', fontsize=8, color='black')

#     plt.tight_layout(pad=5.0)
#     plt.subplots_adjust(hspace=0.2, top=0.90)
#     plt.show()
#     #  Save the plot to a buffer
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     return data

from matplotlib.patches import Rectangle
import math  # for checking infinity


# def plot_yearly_percentage_changes(df, years):

#     if len(years) < 2:
#         return None
#     # Convert RESPONSE_TIMESTAMP to datetime and extract year and month
#     df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'], utc=True)
#     df['Year'] = df['RESPONSE_TIMESTAMP'].dt.year
#     df['Month'] = df['RESPONSE_TIMESTAMP'].dt.month

#     # Filter for the years of interest
#     df = df[df['Year'].isin(years)]

#     # Calculate month-wise sentiment count
#     monthly_counts = df.groupby(['Year', 'Month', 'predicted_sentiment']).size().unstack(fill_value=0)

#     # Function to calculate percentage changes between two years
#     def calculate_percentage_change(year1, year2):
#         percentage_change_data = {'Month': []}
        
#         for month in range(1, 13):
#             if month in monthly_counts.loc[year1].index and month in monthly_counts.loc[year2].index:
#                 change = (monthly_counts.loc[year2].loc[month] - monthly_counts.loc[year1].loc[month]) / monthly_counts.loc[year1].loc[month] * 100
#                 percentage_change_data['Month'].append(f'{month:02d}-{year1}-{year2}')
#                 for sentiment in change.index:
#                     if sentiment not in percentage_change_data:
#                         percentage_change_data[sentiment] = []
#                     percentage_change_data[sentiment].append(change[sentiment])

#         return pd.DataFrame(percentage_change_data).set_index('Month')

#     # Calculate percentage changes for the given years
#     data_frames = []
#     titles = []
#     for i in range(len(years) - 1):
#         year1, year2 = years[i], years[i + 1]
#         data_frames.append(calculate_percentage_change(year1, year2))
#         titles.append(f'% Change Between Year {year1} & {year2}')

#     # Plot configuration
#     sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
#     color_mapping = {'NEGATIVE': 'red', 'NEUTRAL': 'blue', 'POSITIVE': 'green'}

#     # Calculate required rows and columns based on the number of plots
#     num_plots = len(data_frames) * len(sentiments)
#     nrows = (num_plots // 3) + (num_plots % 3 > 0)  # 3 columns, dynamically calculate rows
#     ncols = 3
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 6))
#     axes = axes.flatten()

#     # Plot each sentiment for the year combinations
#     for plot_index, (data_frame, title) in enumerate(zip(data_frames, titles)):
#         for i, sentiment in enumerate(sentiments):
#             ax_index = plot_index * len(sentiments) + i
#             if sentiment in data_frame.columns:
#                 axes[ax_index].plot(data_frame.index, data_frame[sentiment], marker='o', 
#                                     label=sentiment, color=color_mapping.get(sentiment, 'blue'))
                
#                 # Annotations for percentage changes
#                 # for j in range(len(data_frame)):
#                 #     axes[ax_index].annotate(f"{data_frame[sentiment].iloc[j]:.2f}%", 
#                 #                             (data_frame.index[j], data_frame[sentiment].iloc[j]), 
#                 #                             textcoords="offset points", 
#                 #                             xytext=(0, 5), 
#                 #                             ha='center', 
#                 #                             fontsize=8, 
#                 #                             color=axes[ax_index].get_lines()[0].get_color(),
#                 #                             rotation=45)
#                 for j in range(len(data_frame)):
#                     # axes[ax_index].annotate(f"{round(data_frame[sentiment].iloc[j]):.0f}%", 
#                     #                         (data_frame.index[j], data_frame[sentiment].iloc[j]), 
#                     #                         textcoords="offset points", 
#                     #                         xytext=(0, 5), 
#                     #                         ha='center', 
#                     #                         fontsize=12, 
#                     #                         color=axes[ax_index].get_lines()[0].get_color(),
#                     #                         rotation=45)
                    
#                     sentiment_value = data_frame[sentiment].iloc[j]
    
#     # Skip if the sentiment value is infinite or NaN
#                     if math.isinf(sentiment_value) or math.isnan(sentiment_value):
#                         continue
                    
#                     axes[ax_index].annotate(
#                         f"{round(sentiment_value):.0f}%",  # Formatting as percentage
#                         (data_frame.index[j], sentiment_value),  # Position of the annotation
#                         textcoords="offset points",  # Using offset for position
#                         xytext=(0, 5),  # Offset of 5 points along y-axis
#                         ha='center',  # Horizontal alignment at center
#                         fontsize=12,  # Font size of the annotation
#                         color=axes[ax_index].get_lines()[0].get_color(),  # Match line color
#                         rotation=45  # Rotation of text for readability
#     )
                
#                 if sentiment == 'NEUTRAL':
#                     axes[ax_index].set_title(title, fontsize=12,fontweight='bold',pad=22)
                
#                 axes[ax_index].set_ylabel('% Change in Sentiment Count', fontsize=12)
#                 axes[ax_index].axhline(0, color='grey', linewidth=0.8, linestyle='--')
#                 axes[ax_index].grid()
#                 axes[ax_index].legend(title='Sentiment', fontsize='small')
#                 axes[ax_index].set_xticks([])

#                 # Add month labels below the plot
#                 # for j in range(12):
#                 #     month_number = j + 1
#                 #     axes[ax_index].text(j, -5, str(month_number), ha='left', fontsize=8, color='black')

                
#                 # Add month labels inside the plot area with dynamic positioning
#                 for j in range(12):
#                     month_number = j + 1
#                     y_position = axes[ax_index].get_ylim()[0] + 0.05 * (axes[ax_index].get_ylim()[1] - axes[ax_index].get_ylim()[0])
#                     axes[ax_index].text(
#                         j,  # X-coordinate for each month
#                         y_position,  # Adjust Y-coordinate based on the plot's Y-axis lower limit
#                         str(month_number),
#                         ha='center',
#                         fontsize=8,
#                         color='black'
#                     )


#                 # Adjust the month labels to appear inside the plot area
#                 # for j in range(12):
#                 #     month_number = j + 1
#                 #     axes[ax_index].text(
#                 #         j,  # X-coordinate for each month
#                 #         axes[ax_index].get_ylim()[0] + 10,  # Adjust Y-coordinate based on the plot's Y-axis lower limit
#                 #         str(month_number), 
#                 #         ha='center', 
#                 #         fontsize=8, 
#                 #         color='black'
#                 #     )


#                 # Add a solid black frame around each plot (subplot)
#                 for spine in axes[ax_index].spines.values():
#                     spine.set_edgecolor('black')
#                     spine.set_linewidth(2)

#     for plot_index in range(len(data_frames)):
#         # Get the position of the group (x, y, width, height)
#         x = 0  # Start of the group (always from the first column)
#         y = plot_index * 3  # Each group is 3 rows
#         width = 3  # 3 columns (POSITIVE, NEUTRAL, NEGATIVE)
#         height = 3  # 3 rows (if needed, adjust for non-full plots)

#         # Add a rectangle around the group of plots for this year combination
#         rect = Rectangle((x, y), width, height, linewidth=3, edgecolor='black', facecolor='none')
#         fig.patches.append(rect)

#     # Remove extra subplots if not needed
#     for i in range(num_plots, len(axes)):
#         fig.delaxes(axes[i])

#     plt.tight_layout(pad=5.0)
#     plt.subplots_adjust(hspace=0.4, top=0.90)
#     plt.show()

#     # Save the plot to a buffer
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     return data


def plot_yearly_percentage_changes(df, years):
    if len(years) < 2:
        return None
    # Convert RESPONSE_TIMESTAMP to datetime and extract year and month
    df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'], utc=True)
    df['Year'] = df['RESPONSE_TIMESTAMP'].dt.year
    df['Month'] = df['RESPONSE_TIMESTAMP'].dt.month

    # Filter for the years of interest
    df = df[df['Year'].isin(years)]

    # Calculate month-wise sentiment count
    monthly_counts = df.groupby(['Year', 'Month', 'predicted_sentiment']).size().unstack(fill_value=0)

    # Function to calculate percentage changes between two years
    def calculate_percentage_change(year1, year2):
        percentage_change_data = {'Month': []}
        
        for month in range(1, 13):
            if month in monthly_counts.loc[year1].index and month in monthly_counts.loc[year2].index:
                change = (monthly_counts.loc[year2].loc[month] - monthly_counts.loc[year1].loc[month]) / monthly_counts.loc[year1].loc[month] * 100
                percentage_change_data['Month'].append(f'{month:02d}-{year1}-{year2}')
                for sentiment in change.index:
                    if sentiment not in percentage_change_data:
                        percentage_change_data[sentiment] = []
                    percentage_change_data[sentiment].append(change[sentiment])

        return pd.DataFrame(percentage_change_data).set_index('Month')

    # Calculate percentage changes for the given years
    data_frames = []
    titles = []
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        data_frames.append(calculate_percentage_change(year1, year2))
        titles.append(f'% Change Between Year {year1} & {year2}')

    # Plot configuration
    sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    color_mapping = {'NEGATIVE': 'red', 'NEUTRAL': 'blue', 'POSITIVE': 'green'}

    # Calculate required rows and columns based on the number of plots
    num_plots = len(data_frames) * len(sentiments)
    nrows = (num_plots // 3) + (num_plots % 3 > 0)  # 3 columns, dynamically calculate rows
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    # # Plot each sentiment for the year combinations
    # for plot_index, (data_frame, title) in enumerate(zip(data_frames, titles)):
    #     for i, sentiment in enumerate(sentiments):
    #         ax_index = plot_index * len(sentiments) + i
    #         if sentiment in data_frame.columns:
    #             axes[ax_index].plot(data_frame.index, data_frame[sentiment], marker='o', 
    #                                 label=sentiment, color=color_mapping.get(sentiment, 'blue'))
                
    #             # Annotations for percentage changes
    #             for j in range(len(data_frame)):
    #                 sentiment_value = data_frame[sentiment].iloc[j]
    
    #                 if math.isinf(sentiment_value) or math.isnan(sentiment_value):
    #                     continue
                    
    #                 axes[ax_index].annotate(
    #                     f"{round(sentiment_value):.0f}%",  # Formatting as percentage
    #                     (data_frame.index[j], sentiment_value),  # Position of the annotation
    #                     textcoords="offset points",  # Using offset for position
    #                     xytext=(0, 5),  # Offset of 5 points along y-axis
    #                     ha='center',  # Horizontal alignment at center
    #                     fontsize=12,  # Font size of the annotation
    #                     color=axes[ax_index].get_lines()[0].get_color(),  # Match line color
    #                     rotation=45  # Rotation of text for readability
    #                 )
                
    #             if sentiment == 'NEUTRAL':
    #                 axes[ax_index].set_title(title, fontsize=12, fontweight='bold', pad=22)
                
    #             axes[ax_index].set_ylabel('% Change in Sentiment Count', fontsize=12)
    #             axes[ax_index].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    #             axes[ax_index].grid()
    #             axes[ax_index].legend(title='Sentiment', fontsize='small')
    #             axes[ax_index].set_xticks([])

    #             # Add month labels inside the plot area with dynamic positioning
    #             # Ensure month numbers appear based on available data
    #             # for j, month in enumerate(data_frame.index):
    #             #     y_position = axes[ax_index].get_ylim()[0] + 0.05 * (axes[ax_index].get_ylim()[1] - axes[ax_index].get_ylim()[0])
    #             #     axes[ax_index].text(
    #             #         j,  # X-coordinate for each month
    #             #         y_position,  # Adjust Y-coordinate based on the plot's Y-axis lower limit
    #             #         str(month),  # Display the actual month number (1-12)
    #             #         ha='center',
    #             #         fontsize=8,
    #             #         color='black'
    #             #     )
    #             for j, month in enumerate(data_frame.index):
    #                 # Adjust the Y-coordinate based on the plot's Y-axis lower limit
    #                 month_n = j+1
    #                 y_position = axes[ax_index].get_ylim()[0] + 0.05 * (axes[ax_index].get_ylim()[1] - axes[ax_index].get_ylim()[0])

    #                 # Display the month number at the correct position
    #                 axes[ax_index].text(
    #                     j,  # X-coordinate for each month (index of the month)
    #                     y_position,  # Y-coordinate (dynamically adjusted)
    #                     f"{month_n}",  # Just the month number (1-12)
    #                     ha='center',  # Horizontal alignment
    #                     fontsize=8,  # Font size
    #                     color='black'  # Color of the text
    #                 )


    #             # Add a solid black frame around each plot (subplot)
    #             for spine in axes[ax_index].spines.values():
    #                 spine.set_edgecolor('black')
    #                 spine.set_linewidth(2)

    for plot_index, (data_frame, title) in enumerate(zip(data_frames, titles)):
        for i, sentiment in enumerate(sentiments):
            ax_index = plot_index * len(sentiments) + i
            if sentiment in data_frame.columns:
                axes[ax_index].plot(data_frame.index, data_frame[sentiment], marker='o', 
                                    label=sentiment, color=color_mapping.get(sentiment, 'blue'))
                
                # Annotations for percentage changes and month numbers
                for j, month in enumerate(data_frame.index):
                    sentiment_value = data_frame[sentiment].iloc[j]
                    
                    # Skip if sentiment value is infinite or NaN
                    if math.isinf(sentiment_value) or math.isnan(sentiment_value):
                        continue
                    
                    # Annotate percentage value
                    axes[ax_index].annotate(
                        f"{round(sentiment_value):.0f}%",  # Formatting as percentage
                        (data_frame.index[j], sentiment_value),  # Position of the annotation
                        textcoords="offset points",  # Using offset for position
                        xytext=(0, 5),  # Offset of 5 points along y-axis
                        ha='center',  # Horizontal alignment at center
                        fontsize=12,  # Font size of the annotation
                        color=axes[ax_index].get_lines()[0].get_color(),  # Match line color
                        rotation=45  # Rotation of text for readability
                    )
                    
                    # Only show month number if sentiment value is valid
                    month_number = j+1 # Extract the numeric month (1-12)
                    y_position = axes[ax_index].get_ylim()[0] + 0.05 * (axes[ax_index].get_ylim()[1] - axes[ax_index].get_ylim()[0])

                    # Annotate the month number only for valid months
                    axes[ax_index].text(
                        j,  # X-coordinate for each month (index of the month)
                        y_position,  # Y-coordinate (dynamically adjusted)
                        f"{month_number}",  # Just the month number (1-12)
                        ha='center',  # Horizontal alignment
                        fontsize=12,  # Font size
                        color='black'  # Color of the text
                    )
                    
                
                    axes[ax_index].set_ylabel('% Change in Sentiment Count', fontsize=12)
                    axes[ax_index].axhline(0, color='grey', linewidth=0.8, linestyle='--')
                    axes[ax_index].grid()
                    axes[ax_index].legend(title='Sentiment', fontsize='small')
                    axes[ax_index].set_xticks([])

                    # Add a solid black frame around each plot (subplot)
                for spine in axes[ax_index].spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(2)

            if sentiment == 'NEUTRAL':
                        axes[ax_index].set_title(title, fontsize=12, fontweight='bold', pad=22)

    for plot_index in range(len(data_frames)):
        # Get the position of the group (x, y, width, height)
        x = 0  # Start of the group (always from the first column)
        y = plot_index * 3  # Each group is 3 rows
        width = 3  # 3 columns (POSITIVE, NEUTRAL, NEGATIVE)
        height = 3  # 3 rows (if needed, adjust for non-full plots)

        # Add a rectangle around the group of plots for this year combination
        # rect = Rectangle((x, y), width, height, linewidth=3, edgecolor='black', facecolor='none')
        # fig.patches.append(rect)

    # Remove extra subplots if not needed
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.subplots_adjust(hspace=0.4, top=0.90)
    plt.tight_layout(pad=5.0)
    
    plt.show()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return data


from io import StringIO

from django.contrib import messages


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main Index Function  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# def index(request):
#     form = FileUploadForm(request.POST, request.FILES if request.method == 'POST' else None)

    
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#     import re
#     import pandas as pd
#     from nltk.corpus import stopwords

#     from tqdm import tqdm 
#     from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
#     context = {'form': form}  

#     if request.method == 'POST':

        

#         # form = FileUploadForm(request.POST, request.FILES)
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
            
#             # file_name = excel_file.name.lower()

#             # # Check if the file is CSV or Excel
#             # if file_name.endswith('.csv'):
#             #     # Read directly if CSV
#             #     df = pd.read_csv(excel_file)
#             # elif file_name.endswith(('.xls', '.xlsx')):
#             #     # Convert Excel to CSV in memory
#             #     df_excel = pd.read_excel(excel_file)
#             #     csv_buffer = StringIO()
#             #     df_excel.to_csv(csv_buffer, index=False)
#             #     csv_buffer.seek(0)  # Move to the start of the StringIO object
#             #     df = pd.read_csv(csv_buffer)
#             # else:
#             #     # Show a popup message for unsupported file types
#             #     messages.error(request, "Please upload a file in CSV format.")
#             #     return render(request, 'index.html', context)
            
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

#                 # if sentiment_counts:  # Check if sentiment_counts is not empty
#                 #     most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
#                 # else:
#                 #     most_frequent_sentiment = "neutral"  # Default sentiment for empty dictionary

#                 # return most_frequent_sentiment
#                 if sentiment_counts:
#                     positive_count = sentiment_counts.get("POSITIVE", 0)
#                     negative_count = sentiment_counts.get("NEGATIVE", 0)

#                     if positive_count == negative_count:
#                         return "NEUTRAL"  # Equal positive and negative counts
#                     elif positive_count > negative_count:
#                         return "POSITIVE"
#                     else:
#                         return "NEGATIVE"
#                 else:
#                     return "NEUTRAL" 

#                         # # Load the CSV file
#                         # print(excel_file)


#             df = pd.read_csv(excel_file)




#             global df_universal  
#             df_universal = df


#             # plot_data_image = update_plot(request, df)



#             df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'],format='mixed',utc=True)
#             # df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'], utc=True)  # Convert to UTC datetime
#             # df['RESPONSE_TIMESTAMP'] = df['RESPONSE_TIMESTAMP'].apply(
#             #     lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' Z' if pd.notnull(x) else None
#             # )



#             # years = df['RESPONSE_TIMESTAMP'].dt.year.unique()
#             years = sorted(df['RESPONSE_TIMESTAMP'].dt.year.unique())
#             clean_years = [int(year) for year in years if not pd.isna(year)]
#             years = clean_years
#             # years = sorted(df['RESPONSE_TIMESTAMP'].dt.year.dropna().unique().astype(int))

#             # years = [int(float(year)) for year in years.split(',') if year and year.lower() != 'nan']

#             print(years)
            
#             # print(year_options)
#             # years = (request.POST.get('year_options'))
#             # years=[2016,2017]
#             # year = year_options[years]   
#             # selected_year = years[0] if years.size > 0 else None  
#             # Sort years in ascending order and select the first one if available
#             selected_year = sorted(years)[0] if len(years) > 0 else None

#             # # ... (rest of your update_plot function)
#             # plot_data_image = update_plot(year,df)
#             # return JsonResponse({'plot_data': plot_data})

#             # df = df[(df['review'] != "undefined")].head(100)

#             # print(df)

#             # Load the stopwords set

#             stopwords_set = set(stopwords.words('english'))

#             # Clean the reviews, convert to lowercase, and remove stopwords

            

# # Filter words based on sentiment and get frequency counts
#             # positive_words = [word for word, count in word_counts.items() if TextBlob(word).sentiment.polarity > 0]
#             # negative_words = [word for word, count in word_counts.items() if TextBlob(word).sentiment.polarity < 0]

#             # positive_word_counts = Counter(positive_words)
#             # negative_word_counts = Counter(negative_words)

#             # # Get the top 10 most frequent positive and negative sentiment words
#             # top_10_positive_words = positive_word_counts.most_common(30)
#             # top_10_negative_words = negative_word_counts.most_common(30)

#             # print("Top 10 most frequent positive sentiment words:")
#             # print(top_10_positive_words)

#             # print("Top 10 most frequent negative sentiment words:")
#             # print(top_10_negative_words)



            

#             # Total_review = 2000
#             # print("Total_review",Total_review)

           


#             total_reviews = len(df)
#             pbar = tqdm(total=total_reviews)
#              # Initialize progress percentage
#             progress_percentage = 0

#             ground_truth = []
#             predictions = []

#             # Iterate through each row and analyze sentiment
#             if 'predicted_sentiment' not in df.columns:
#                 df['predicted_sentiment'] = None

#             sentiment_map = {'POSITIVE': 'Promoter', 'NEUTRAL': 'Passive', 'NEGATIVE': 'Detractor'}

#             # for index, row in df.iterrows():
#             #     review_text = str(row['review'])
#             #     predicted_sentiment = predict_sentiment(review_text)

#             #     # print(predicted_sentiment)
                
#             #     df.at[index, 'predicted_sentiment'] = predicted_sentiment

#                 # if df.at[index,'review'] == 'undefined':
#                 #     df.at[index, 'predicted_sentiment'] = "NEUTRAL"

#                 # if pd.isna(review_text):
#                 #     print("yess")
#                 #     df.at[index, 'review'] = "Positive review"
#                 #     df.at[index, 'predicted_sentiment'] = "POSITIVE"
#                 #     df.at[index, 'nps_category'] = "Promoter"
                
#                 # # review_text = str(df.loc[index, 'review'])
#                 # if review_text and review_text.strip() and any(char.isalnum() for char in review_text):
#                 #     for predicted_sentiment in df['predicted_sentiment']:
#                 #         predicted_sentiment = df.loc[index, 'predicted_sentiment']
#                 #         if predicted_sentiment in sentiment_map:
#                 #             df.at[index, 'nps_category'] = sentiment_map[predicted_sentiment]


#                 # if all(not char.isalnum() for char in review_text):
#                 #     score = row['SCORE']
#                 #     if 0 <= score <= 6:
#                 #         df.at[index, 'nps_category'] = "Detractor"
#                 #         df.at[index, 'predicted_sentiment'] = "NEGATIVE"
#                 #     elif 7 <= score <= 8:
#                 #         df.at[index, 'nps_category'] = "Neutral"
#                 #         df.at[index, 'predicted_sentiment'] = "NEUTRAL"
#                 #     elif 9 <= score <= 10:
#                 #         df.at[index, 'nps_category'] = "Promoter"
#                 #         df.at[index, 'predicted_sentiment'] = "POSITIVE"
#                 #     else:
#                 #         df.at[index, 'nps_category'] = "NA"


                
#                 # ground_truth.append(row['predicted_sentiment'])  # Assuming 'sentiment' column holds actual labels
#                 # predictions.append(predicted_sentiment)

#                 # Update progress bar and display percentage
#             #     pbar.update(1)
#             # completed = (index + 1) / total_reviews * 100
#             # progress_percentage = int(completed)    
#             # pbar.set_description(f"Progress: {completed:.2f}%")
#                 # completed = (index + 1) / total_reviews * 100
#                 # pbar.update(1)
#                 # pbar.set_description(f"Sentiment Prediction Progress: {completed:.2f}%")

#             # df.to_csv('YES.csv', index=False)
#             # df['clean_review'] = df['review'].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word != 'undefined' and word.isalpha() and word not in stopwords_set]))
#             # # Get word frequency counts


#             # negative_reviews = df[df['predicted_sentiment'] == 'NEGATIVE']['clean_review']

#             # # Get word frequency counts for negative reviews
#             # non_empty_reviews = negative_reviews[negative_reviews != '']
#             # negative_word_counts = Counter(non_empty_reviews.str.split().sum())
            

#             # # Get the top 10 most frequent negative sentiment words
#             # top_10_negative_words = sorted(negative_word_counts.most_common(10), key=lambda x: x[1], reverse=True)


#             # word_counts = Counter(df['clean_review'].str.split().sum())

#             # from textblob import TextBlob
            

#             # top_10_words = sorted(word_counts.most_common(10), key=lambda x: x[1], reverse=True)

           

#             # least_10_words = top_10_negative_words


#             # print(df.columns)

#             # df = df[(df['review'] != "undefined")] 

#             if 'predicted_sentiment' in df.columns:
                    
#                     print("okokokokokokokokok")
#                     positive_count = df[df['predicted_sentiment'] == 'POSITIVE'].shape[0]
#                     negative_count = df[df['predicted_sentiment'] == 'NEGATIVE'].shape[0]
#                     neutral_count = df[df['predicted_sentiment'] == 'NEUTRAL'].shape[0]
#                     print('positive_count',positive_count)
#                     print('negative_count',negative_count)
#                     print('neutral_count',neutral_count)
#                     Total_review = positive_count + negative_count + neutral_count

#                     # Calculate percentages using integer division
#                     positive_count_percentage = (positive_count * 100) / Total_review
#                     negative_count_percentage = (negative_count * 100) / Total_review
#                     neutral_count_percentage = (neutral_count * 100) / Total_review

#                     # Format percentages for display
#                     positive_count_percentage = "{:.2f}".format(positive_count_percentage)
#                     negative_count_percentage = "{:.2f}".format(negative_count_percentage)
#                     neutral_count_percentage = "{:.2f}".format(neutral_count_percentage)

#                     print("positive_count_percentage:", positive_count_percentage)
#                     print("negative_count_percentage:", negative_count_percentage)
#                     print("neutral_count_percentage:", neutral_count_percentage)
#                     show_results = True
#             else:
#                     # Handle the case where the 'sentimental_analysis' column doesn't exist
#                 pass

#             # # Add a new column for predicted sentiment (optional)
#             # if 'predicted_sentiment' not in df.columns:
#             #     df['predicted_sentiment'] = None

#             pbar.close()  

#             # Save the updated DataFrame to a CSV file (specify the output path)
#             # df.to_csv(r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv', index=False)

#             # total_reviews = len(df)
#             # pbar = tqdm(total=total_reviews, desc='Analyzing Reviews')

#             # Print the DataFrame with the added column (optional)
#             # print(df.head(50))

#             # plot_data = create_plot(df)
#             # negative_plot_data = get_bottom_5_negative_suppliers(df)
#             # product_plot_data = create_positive_sentiment_plot(df)
#             # product_negative_plot_data = create_negative_sentiment_plot(df)
#             # word_cloud = generate_sentiment_wordclouds(df)
#             # Count_of_keywords_in_each_depart=Count_of_keywords_in_each_department(df)
#             # word_cloud_Fedex = generate_sentiment_wordclouds_for_fedex(df)
#             # NPSScoress=NPSScore(df)
#             # NPSThree=NetPro(df)
             
            

#             # year_dropdown = create_year_dropdown(df)    

#             # ProductQualityPositiveKey =ProductQualityPositiveKeyword(df)
#             # ProductQualityNegativeKey = ProductQualityNegativeKeyword(df)
#             # CustomerServicePositiveKey=CustomerServicePositiveKeyword(df)
#             # CustomerServiceNegativeKey = CustomerServiceNegativeKeyword(df)
#             # PurchaseExperiencePositiveKey=PurchaseExperiencePositiveKeyword(df)
#             # PurchaseExperiencePNegativeKey=PurchaseExperienceNegativeKeyword(df)
#             # PricingPositiveKey=PricingPositiveKeyword(df)
#             # PricingNegativeKey=PricingNegativeKeyword(df)
#             # DeliveryPositiveKey=DeliveryPositiveKeyword(df)
#             # DeliveryNegativeKey=DeliveryNegativeKeyword(df)
            
#             # html_table = generate_html_table(bigram_results)

#             # bigram_results = generate_bigram_analysis(df)

#             # print(bigram_results)





#             # negative_plot_data = get_bottom_5_negative_suppliers()  # Assuming it doesn't return None
#             # context = {'form': form,
#             #    'positive_count': positive_count,
#             #    'negative_count': negative_count,
#             #    'neutral_count': neutral_count,
#             #    'show_results': show_results,
#             #    'top_5_suppliers': top_5_suppliers,
#             #    "positive_count_percentage": positive_count_percentage,
#             #    "negative_count_percentage": negative_count_percentage,
#             #    "neutral_count_percentage": neutral_count_percentage,
#             #    'plot_data': plot_data,
#             #    'word_cloud':word_cloud,
#             #    'word_cloud_Fedex':word_cloud_Fedex,
#             #    'negative_plot_data':negative_plot_data,
#             #    'product_plot_data':product_plot_data,
#             #    'product_negative_plot_data':product_negative_plot_data,
#             #    'progress_percentage': progress_percentage,
#             #    'total_data': total_reviews,
#             #    'least_10_words': least_10_words,
#             #    'top_10_words': top_10_words,
#             #    'PieChartPN_plot':PieChartPN,
#             #    'Count_of_keywords_in_each_department':Count_of_keywords_in_each_depart,
#             #    'NPSScoress':NPSScoress,
#             #    'NPSSS':NPSThree
                             
#             #      # Add negative plot data if available
#             #    }
            

#             context = {'form': form,
#                 'total_data': total_reviews,
#                'positive_count': positive_count,
#                'negative_count': negative_count,
#                'neutral_count': neutral_count,
#                'show_results': show_results,
#                'top_5_suppliers': top_5_suppliers,
#                "positive_count_percentage": positive_count_percentage,
#                "negative_count_percentage": negative_count_percentage,
#                "neutral_count_percentage": neutral_count_percentage,
#                'years':years,
#             #    'bigram_results':bigram_results,
               
#             #    'year_dropdown': year_dropdown,
#             #    'plot_data': update_plot(year_dropdown.value),
#             #    'plot_data_image':plot_data_image
#                'selected_year':selected_year
#             #    'ProductQualityPositiveKey':ProductQualityPositiveKey,
#             #    'ProductQualityNegativeKey':ProductQualityNegativeKey,
#             #    'CustomerServicePositiveKey':CustomerServicePositiveKey,
#             #    'CustomerServiceNegativeKey':CustomerServiceNegativeKey,
#             #    'PurchaseExperiencePositiveKey':PurchaseExperiencePositiveKey,
#             #    'PurchaseExperiencePNegativeKey':PurchaseExperiencePNegativeKey,
#             #    'PricingPositiveKey':PricingPositiveKey,
#             #    'PricingNegativeKey':PricingNegativeKey,
#             #    'DeliveryPositiveKey':DeliveryPositiveKey,
#             #    'DeliveryNegativeKey':DeliveryNegativeKey,
#             #    'PieChartPN':PieChartPN,
#             #    'plot_negative_reviews_par':plot_negative_reviews_par

#             #    'NPSScoress':NPSScoress
              
#                  # Add negative plot data if available
#                }
#             # print("context",context)
#             return render(request, 'index.html', context)
#     else:
#     # Handle GET requests (e.g., initial page load)
#         context = {'form': form}
#         return render(request, 'index.html', context)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def load_departments_for_industry(industry_name):
    with open(r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json", "r") as f:
        config = json.load(f)
    
    # Return the departments for the selected industry
    return config["industries"].get(industry_name, {}).get("departments", [])


def index(request):

    global selected_industry  # Declare as global to assign it

    # industries = load_industries_from_config()

    # Load industry names from config file
    with open(r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json", "r") as f:
        config = json.load(f)
    industries = list(config["industries"].keys()) 



    print('industries',industries)

    form = FileUploadForm(request.POST, request.FILES if request.method == 'POST' else None)

    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import re
    import pandas as pd
    from nltk.corpus import stopwords

    from tqdm import tqdm 
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
    # context = {'form': form}  
    context = {'form': form, 'industries': industries}

    selected_industry = None
    if request.method == 'POST':

        selected_industry = request.POST.get("industry")
        print('selected_industry',selected_industry)

        

        # form = FileUploadForm(request.POST, request.FILES)
        positive_count = 0  # Initialize variables outside the 'if' block
        negative_count = 0
        neutral_count = 0
        show_results = False
        Total_review = 0
        positive_count_percentage=0
        negative_count_percentage=0
        neutral_count_percentage=0


        if form.is_valid():
            excel_file = request.FILES['excel_file']

            
            
            # file_name = excel_file.name.lower()

            # # Check if the file is CSV or Excel
            # if file_name.endswith('.csv'):
            #     # Read directly if CSV
            #     df = pd.read_csv(excel_file)
            # elif file_name.endswith(('.xls', '.xlsx')):
            #     # Convert Excel to CSV in memory
            #     df_excel = pd.read_excel(excel_file)
            #     csv_buffer = StringIO()
            #     df_excel.to_csv(csv_buffer, index=False)
            #     csv_buffer.seek(0)  # Move to the start of the StringIO object
            #     df = pd.read_csv(csv_buffer)
            # else:
            #     # Show a popup message for unsupported file types
            #     messages.error(request, "Please upload a file in CSV format.")
            #     return render(request, 'index.html', context)
            
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

                # if sentiment_counts:  # Check if sentiment_counts is not empty
                #     most_frequent_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                # else:
                #     most_frequent_sentiment = "neutral"  # Default sentiment for empty dictionary

                # return most_frequent_sentiment
                if sentiment_counts:
                    positive_count = sentiment_counts.get("POSITIVE", 0)
                    negative_count = sentiment_counts.get("NEGATIVE", 0)

                    if positive_count == negative_count:
                        return "NEUTRAL"  # Equal positive and negative counts
                    elif positive_count > negative_count:
                        return "POSITIVE"
                    else:
                        return "NEGATIVE"
                else:
                    return "NEUTRAL" 

                        # # Load the CSV file
                        # print(excel_file)


            df = pd.read_csv(excel_file)




            global df_universal  
            df_universal = df


            # plot_data_image = update_plot(request, df)



            df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'],format='mixed',utc=True)
            # df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'], utc=True)  # Convert to UTC datetime
            # df['RESPONSE_TIMESTAMP'] = df['RESPONSE_TIMESTAMP'].apply(
            #     lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' Z' if pd.notnull(x) else None
            # )



            # years = df['RESPONSE_TIMESTAMP'].dt.year.unique()
            years = sorted(df['RESPONSE_TIMESTAMP'].dt.year.unique())
            clean_years = [int(year) for year in years if not pd.isna(year)]
            years = clean_years
            # years = sorted(df['RESPONSE_TIMESTAMP'].dt.year.dropna().unique().astype(int))

            # years = [int(float(year)) for year in years.split(',') if year and year.lower() != 'nan']

            print(years)
            
            selected_year = sorted(years)[0] if len(years) > 0 else None

           
            # Load the stopwords set

            stopwords_set = set(stopwords.words('english'))

            # Clean the reviews, convert to lowercase, and remove stopwords

            

# Filter words based on sentiment and get frequency counts
            # positive_words = [word for word, count in word_counts.items() if TextBlob(word).sentiment.polarity > 0]
            # negative_words = [word for word, count in word_counts.items() if TextBlob(word).sentiment.polarity < 0]

            # positive_word_counts = Counter(positive_words)
            # negative_word_counts = Counter(negative_words)

            # # Get the top 10 most frequent positive and negative sentiment words
            # top_10_positive_words = positive_word_counts.most_common(30)
            # top_10_negative_words = negative_word_counts.most_common(30)

            # print("Top 10 most frequent positive sentiment words:")
            # print(top_10_positive_words)

            # print("Top 10 most frequent negative sentiment words:")
            # print(top_10_negative_words)



            

            # Total_review = 2000
            # print("Total_review",Total_review)

           


            total_reviews = len(df)
            pbar = tqdm(total=total_reviews)
             # Initialize progress percentage
            progress_percentage = 0

            ground_truth = []
            predictions = []

            # Iterate through each row and analyze sentiment
            if 'predicted_sentiment' not in df.columns:
                df['predicted_sentiment'] = None

            sentiment_map = {'POSITIVE': 'Promoter', 'NEUTRAL': 'Passive', 'NEGATIVE': 'Detractor'}

            # for index, row in df.iterrows():
            #     review_text = str(row['review'])
            #     predicted_sentiment = predict_sentiment(review_text)

            #     # print(predicted_sentiment)
                
            #     df.at[index, 'predicted_sentiment'] = predicted_sentiment

                # if df.at[index,'review'] == 'undefined':
                #     df.at[index, 'predicted_sentiment'] = "NEUTRAL"

                # if pd.isna(review_text):
                #     print("yess")
                #     df.at[index, 'review'] = "Positive review"
                #     df.at[index, 'predicted_sentiment'] = "POSITIVE"
                #     df.at[index, 'nps_category'] = "Promoter"
                
                # # review_text = str(df.loc[index, 'review'])
                # if review_text and review_text.strip() and any(char.isalnum() for char in review_text):
                #     for predicted_sentiment in df['predicted_sentiment']:
                #         predicted_sentiment = df.loc[index, 'predicted_sentiment']
                #         if predicted_sentiment in sentiment_map:
                #             df.at[index, 'nps_category'] = sentiment_map[predicted_sentiment]


                # if all(not char.isalnum() for char in review_text):
                #     score = row['SCORE']
                #     if 0 <= score <= 6:
                #         df.at[index, 'nps_category'] = "Detractor"
                #         df.at[index, 'predicted_sentiment'] = "NEGATIVE"
                #     elif 7 <= score <= 8:
                #         df.at[index, 'nps_category'] = "Neutral"
                #         df.at[index, 'predicted_sentiment'] = "NEUTRAL"
                #     elif 9 <= score <= 10:
                #         df.at[index, 'nps_category'] = "Promoter"
                #         df.at[index, 'predicted_sentiment'] = "POSITIVE"
                #     else:
                #         df.at[index, 'nps_category'] = "NA"


                
                # ground_truth.append(row['predicted_sentiment'])  # Assuming 'sentiment' column holds actual labels
                # predictions.append(predicted_sentiment)

                # Update progress bar and display percentage
            #     pbar.update(1)
            # completed = (index + 1) / total_reviews * 100
            # progress_percentage = int(completed)    
            # pbar.set_description(f"Progress: {completed:.2f}%")
                # completed = (index + 1) / total_reviews * 100
                # pbar.update(1)
                # pbar.set_description(f"Sentiment Prediction Progress: {completed:.2f}%")

            # df.to_csv('YES.csv', index=False)
            # df['clean_review'] = df['review'].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word != 'undefined' and word.isalpha() and word not in stopwords_set]))
            # # Get word frequency counts


            # negative_reviews = df[df['predicted_sentiment'] == 'NEGATIVE']['clean_review']

            # # Get word frequency counts for negative reviews
            # non_empty_reviews = negative_reviews[negative_reviews != '']
            # negative_word_counts = Counter(non_empty_reviews.str.split().sum())
            

            # # Get the top 10 most frequent negative sentiment words
            # top_10_negative_words = sorted(negative_word_counts.most_common(10), key=lambda x: x[1], reverse=True)


            # word_counts = Counter(df['clean_review'].str.split().sum())

            # from textblob import TextBlob
            

            # top_10_words = sorted(word_counts.most_common(10), key=lambda x: x[1], reverse=True)

           

            # least_10_words = top_10_negative_words


            # print(df.columns)

            # df = df[(df['review'] != "undefined")] 

            if 'predicted_sentiment' in df.columns:
                    
                    print("okokokokokokokokok")
                    positive_count = df[df['predicted_sentiment'] == 'POSITIVE'].shape[0]
                    negative_count = df[df['predicted_sentiment'] == 'NEGATIVE'].shape[0]
                    neutral_count = df[df['predicted_sentiment'] == 'NEUTRAL'].shape[0]
                    print('positive_count',positive_count)
                    print('negative_count',negative_count)
                    print('neutral_count',neutral_count)
                    Total_review = positive_count + negative_count + neutral_count

                    # Calculate percentages using integer division
                    positive_count_percentage = (positive_count * 100) / Total_review
                    negative_count_percentage = (negative_count * 100) / Total_review
                    neutral_count_percentage = (neutral_count * 100) / Total_review

                    # Format percentages for display
                    positive_count_percentage = "{:.2f}".format(positive_count_percentage)
                    negative_count_percentage = "{:.2f}".format(negative_count_percentage)
                    neutral_count_percentage = "{:.2f}".format(neutral_count_percentage)

                    print("positive_count_percentage:", positive_count_percentage)
                    print("negative_count_percentage:", negative_count_percentage)
                    print("neutral_count_percentage:", neutral_count_percentage)
                    show_results = True
            else:
                    # Handle the case where the 'sentimental_analysis' column doesn't exist
                pass

            # # Add a new column for predicted sentiment (optional)
            # if 'predicted_sentiment' not in df.columns:
            #     df['predicted_sentiment'] = None

            pbar.close()  

            # Save the updated DataFrame to a CSV file (specify the output path)
            # df.to_csv(r'C:\Users\Parth Bhavnani\Desktop\sentiment_analyzed.csv', index=False)

            # total_reviews = len(df)
            # pbar = tqdm(total=total_reviews, desc='Analyzing Reviews')

            # Print the DataFrame with the added column (optional)
            # print(df.head(50))

            # plot_data = create_plot(df)
            # negative_plot_data = get_bottom_5_negative_suppliers(df)
            # product_plot_data = create_positive_sentiment_plot(df)
            # product_negative_plot_data = create_negative_sentiment_plot(df)
            # word_cloud = generate_sentiment_wordclouds(df)
            # Count_of_keywords_in_each_depart=Count_of_keywords_in_each_department(df)
            # word_cloud_Fedex = generate_sentiment_wordclouds_for_fedex(df)
            # NPSScoress=NPSScore(df)
            # NPSThree=NetPro(df)
             
            

            # year_dropdown = create_year_dropdown(df)    

            # ProductQualityPositiveKey =ProductQualityPositiveKeyword(df)
            # ProductQualityNegativeKey = ProductQualityNegativeKeyword(df)
            # CustomerServicePositiveKey=CustomerServicePositiveKeyword(df)
            # CustomerServiceNegativeKey = CustomerServiceNegativeKeyword(df)
            # PurchaseExperiencePositiveKey=PurchaseExperiencePositiveKeyword(df)
            # PurchaseExperiencePNegativeKey=PurchaseExperienceNegativeKeyword(df)
            # PricingPositiveKey=PricingPositiveKeyword(df)
            # PricingNegativeKey=PricingNegativeKeyword(df)
            # DeliveryPositiveKey=DeliveryPositiveKeyword(df)
            # DeliveryNegativeKey=DeliveryNegativeKeyword(df)
            
            # html_table = generate_html_table(bigram_results)

            # bigram_results = generate_bigram_analysis(df)

            # print(bigram_results)





            # negative_plot_data = get_bottom_5_negative_suppliers()  # Assuming it doesn't return None
            # context = {'form': form,
            #    'positive_count': positive_count,
            #    'negative_count': negative_count,
            #    'neutral_count': neutral_count,
            #    'show_results': show_results,
            #    'top_5_suppliers': top_5_suppliers,
            #    "positive_count_percentage": positive_count_percentage,
            #    "negative_count_percentage": negative_count_percentage,
            #    "neutral_count_percentage": neutral_count_percentage,
            #    'plot_data': plot_data,
            #    'word_cloud':word_cloud,
            #    'word_cloud_Fedex':word_cloud_Fedex,
            #    'negative_plot_data':negative_plot_data,
            #    'product_plot_data':product_plot_data,
            #    'product_negative_plot_data':product_negative_plot_data,
            #    'progress_percentage': progress_percentage,
            #    'total_data': total_reviews,
            #    'least_10_words': least_10_words,
            #    'top_10_words': top_10_words,
            #    'PieChartPN_plot':PieChartPN,
            #    'Count_of_keywords_in_each_department':Count_of_keywords_in_each_depart,
            #    'NPSScoress':NPSScoress,
            #    'NPSSS':NPSThree
                             
            #      # Add negative plot data if available
            #    }
            

            context = {'form': form,
                'total_data': total_reviews,
               'positive_count': positive_count,
               'negative_count': negative_count,
               'neutral_count': neutral_count,
               'show_results': show_results,
               'top_5_suppliers': top_5_suppliers,
               "positive_count_percentage": positive_count_percentage,
               "negative_count_percentage": negative_count_percentage,
               "neutral_count_percentage": neutral_count_percentage,
               'years':years,
            #    'bigram_results':bigram_results,
               
            #    'year_dropdown': year_dropdown,
            #    'plot_data': update_plot(year_dropdown.value),
            #    'plot_data_image':plot_data_image
                'industries':industries,
                'selected_year':selected_year,
                'selected_industry':selected_industry
               
            #    'ProductQualityPositiveKey':ProductQualityPositiveKey,
            #    'ProductQualityNegativeKey':ProductQualityNegativeKey,
            #    'CustomerServicePositiveKey':CustomerServicePositiveKey,
            #    'CustomerServiceNegativeKey':CustomerServiceNegativeKey,
            #    'PurchaseExperiencePositiveKey':PurchaseExperiencePositiveKey,
            #    'PurchaseExperiencePNegativeKey':PurchaseExperiencePNegativeKey,
            #    'PricingPositiveKey':PricingPositiveKey,
            #    'PricingNegativeKey':PricingNegativeKey,
            #    'DeliveryPositiveKey':DeliveryPositiveKey,
            #    'DeliveryNegativeKey':DeliveryNegativeKey,
            #    'PieChartPN':PieChartPN,
            #    'plot_negative_reviews_par':plot_negative_reviews_par

            #    'NPSScoress':NPSScoress
              
                 # Add negative plot data if available
               }
            # print("context",context)
            return render(request, 'index.html', context)
    else:
    # Handle GET requests (e.g., initial page load)
        context = {'form': form,'industries':industries,'selected_industry':selected_industry}
        return render(request, 'index.html', context)




def convert_timestamp(df):
  df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'],format='mixed',utc=True)
  return df



def calculate_sentiment_counts(df,years):

    df = convert_timestamp(df.copy())
    
    print("Selected years:", years)

    # Filter data for the selected years
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]


    positive_count = filtered_df[filtered_df['predicted_sentiment'] == 'POSITIVE'].shape[0]
    negative_count = filtered_df[filtered_df['predicted_sentiment'] == 'NEGATIVE'].shape[0]
    neutral_count = filtered_df[filtered_df['predicted_sentiment'] == 'NEUTRAL'].shape[0]
    
    print('positive_count', positive_count)
    print('negative_count', negative_count)
    print('neutral_count', neutral_count)
    
    total_review = positive_count + negative_count + neutral_count

    # Avoid division by zero
    if total_review > 0:
        # Calculate percentages
        positive_count_percentage = (positive_count * 100) / total_review
        negative_count_percentage = (negative_count * 100) / total_review
        neutral_count_percentage = (neutral_count * 100) / total_review
    else:
        positive_count_percentage = negative_count_percentage = neutral_count_percentage = 0.0

    # Format percentages for display
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_count_percentage': "{:.2f}".format(positive_count_percentage),
        'negative_count_percentage': "{:.2f}".format(negative_count_percentage),
        'neutral_count_percentage': "{:.2f}".format(neutral_count_percentage)
    }


def update_plot_route(request):
    global df_universal,selected_industry
    
    # print('selected_industry',selected_industry)
    year_str = request.GET.get('years')

    print("year_str",year_str)

    # years = [int(year) for year in year_str.split(',') if year]
    # years = [int(float(year)) for year in year_str.split(',') if year]
    years = [int(float(year)) for year in year_str.split(',') if year and year.lower() != 'nan']


    
    print('years',years)

    try:
        # Convert it to an integer
        # year = int(year_str)
        
        # total_reviews = len(df_universal)
        image_data = update_plot(years,df_universal)
        nps_image  = NPSScore(df_universal, years)
        npspro_image = NetPro(df_universal, years)
        counts_percentage = calculate_sentiment_counts(df_universal,years)
        config_path= r"C:\Users\minal\OneDrive\Desktop\New folder\RushabhMehta\RushabhMehta\Sentimental\Senti\UI_Need\config.json"
        PieChartPN = BarChartPN_plot(df_universal,years,selected_industry,config_path)
        plot_negative_reviews_par=plot_negative_reviews_pareto(df_universal, years,selected_industry)
        plot_bigrams_by_sentime=plot_bigrams_by_sentiment(df_universal,years,selected_industry)
        plot_yearly_percentage=plot_yearly_percentage_changes(df_universal, years)
        
        
        # plot_negative_reviews_par=plot_negative_reviews_pareto(df_universal,years)

        df = convert_timestamp(df_universal.copy())
    
        print("Selected years:", years)

        # Filter data for the selected years
        filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

        # filtered_df = df_universal[df_universal['RESPONSE_TIMESTAMP'].isin(years)]  # Replace 'year_column_name' with the actual column name
        total_reviews = len(filtered_df)

        return JsonResponse({'image': image_data,'nps_image':nps_image,'npspro_image':npspro_image,'counts_percentage':counts_percentage,'total_reviews': total_reviews,'years':years,'PieChartPN':PieChartPN,'plot_negative_reviews_par':plot_negative_reviews_par,'plot_yearly_percentage':plot_yearly_percentage,'plot_bigrams_by_sentime':plot_bigrams_by_sentime})
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except TypeError:
        return JsonResponse({'error': 'Invalid year parameter'}, status=400)

# df = convert_timestamp(df.copy())  # Apply conversion to a copy of df

# Define year options for the dropdown  
def create_year_dropdown(df):
    """Creates a dropdown widget for selecting years from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        widgets.Dropdown: The created dropdown widget.
    """
    df['RESPONSE_TIMESTAMP'] = pd.to_datetime(df['RESPONSE_TIMESTAMP'])
    year_options = df['RESPONSE_TIMESTAMP'].dt.year.unique()

    year_dropdown = widgets.Dropdown(
        options=year_options,
        description="Year:",
        value=year_options.min()
    )

    return year_dropdown

# def update_plot(year,df):
  
#   print("goodddd")
  
#   df = convert_timestamp(df.copy())

#   print(df.head(10))


#   # Filter data for the selected year
# #   filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year == year]

#   year=int(year)


#   filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin([year])]  # Use isin for multiple possible years

#   # Count sentiment values for each month
#   monthly_counts = filtered_df.groupby(pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'))['predicted_sentiment'].value_counts().unstack()

#   # Handle potential missing months (if data is sparse)
#   monthly_counts = monthly_counts.fillna(0)  # Fill missing months with 0 count

#   # Create a line chart
#   monthly_counts.plot(kind='line', figsize=(10, 6))
#   plt.xlabel('Month')
#   plt.ylabel('Count')
#   plt.title(f'Sentiment Trend ({year})')
#   plt.legend(title='Sentiment')
# #   plt.show()

#   # Create image data for potential display (optional)
#   buf = BytesIO()
#   plt.savefig(buf, format='png')
#   plt.close()
#   data = base64.b64encode(buf.getvalue()).decode('utf-8')
#   return data


# def update_plot(years, df):
#     df = convert_timestamp(df.copy())
    
#     print("jji year",years)

#     # Filter data for the selected years
#     filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.isin(years)]

#     # Count sentiment values for each month
#     monthly_counts = filtered_df.groupby(pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'))['predicted_sentiment'].value_counts().unstack()
#     monthly_counts = monthly_counts.fillna(0)

#     # Create a line chart
#     monthly_counts.plot(kind='line', figsize=(10, 6))
#     plt.xlabel('Month')
#     plt.ylabel('Count')
#     plt.title(f'Sentiment Trend ({", ".join(map(str, years))})')
#     plt.legend(title='Sentiment')

#     # Create image data for potential display
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
#     return data


def update_plot(years, df):

    
    df = convert_timestamp(df.copy())
    
    print("Selected years:", years)

    # Filter data for the selected years
    filtered_df = df[df['RESPONSE_TIMESTAMP'].dt.year.astype(int).isin(years)]

    # Check if we are dealing with multiple years
    if len(years) > 1:
        # Sum sentiment values for each month if multiple years are selected
        # monthly_counts = (filtered_df
        #                   .groupby(pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'))['predicted_sentiment']
        #                   .value_counts()  # Sum sentiment values for each month
        #                   .reset_index())
        monthly_counts = (filtered_df
                          .groupby([pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'), 'predicted_sentiment'])
                          .size()  # Count occurrences for each month and sentiment
                          .unstack(fill_value=0)  # Pivot the table to have sentiments as columns
                          .reset_index())


    else:
        # Count sentiment values for the single selected year
        monthly_counts = (filtered_df
                          .groupby(pd.Grouper(key='RESPONSE_TIMESTAMP', freq='M'))['predicted_sentiment']
                          .value_counts()  # Count occurrences for the single year
                          .unstack(fill_value=0)  # Fill missing values with 0
                          .reset_index())

    # Check if the DataFrame is empty
    if monthly_counts.empty:
        print("No data available for the selected years.")
        return None

    # Create a line chart
    plt.figure(figsize=(10, 6))
    
    # Efficient plotting for multiple years
    if len(years) > 1:
        # plt.plot(monthly_counts['RESPONSE_TIMESTAMP'], monthly_counts['predicted_sentiment'], label=str("sentiment"), marker='o')
        for sentiment in monthly_counts.columns:
            if sentiment != 'RESPONSE_TIMESTAMP':  # Skip the timestamp column
                plt.plot(monthly_counts['RESPONSE_TIMESTAMP'], monthly_counts[sentiment], label=str(sentiment), marker='o')
    else:
        for sentiment in monthly_counts.columns[1:]:  # Skip the timestamp column
            plt.plot(monthly_counts['RESPONSE_TIMESTAMP'], monthly_counts[sentiment], label=str(sentiment), marker='o')

    plt.xlabel('Month')
    plt.ylabel('Sentiment Count')
    wrapped_title = wrap_text(f'Sentiment Trend ({", ".join(map(str, years))})', line_length=50)
    plt.title(wrapped_title, fontsize=10, fontweight='bold')
    # plt.title(f'Sentiment Trend ({", ".join(map(str, years))})')
    plt.legend(title='Sentiment')

    # Create image data for potential display
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data

import time

# def generate_progress():
#     total_iterations = 100
#     for i in tqdm(range(total_iterations)):
#         time.sleep(0.1)  # Simulating work
#         progress = (i + 1) / total_iterations * 100
#         yield f"data: {progress}\n\n"  # Sending progress via SSE

# def progress(request):
#     # Set the content type to text/event-stream for SSE
#     response = StreamingHttpResponse(generate_progress(), content_type='text/event-stream')
#     response['Cache-Control'] = 'no-cache'
#     return response


# def progress(request):
#     def event_stream():
#         # Simulating progress from 0% to 100%
#         for i in range(1, 101):
#             time.sleep(0.1)  # Simulate some processing time
#             yield f'data: {i}\n\n'

#     return StreamingHttpResponse(event_stream(), content_type='text/event-stream')