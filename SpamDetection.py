import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv(r"/Users/varunsaravanan/Desktop/SpamDetection/spam_ham_dataset.csv")
data.drop_duplicates(inplace=True)
data['label'] = data['label'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Splitting the dataset
mess = data['text']
cat = data['label']
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
features_train = tfidf.fit_transform(mess_train)
features_test = tfidf.transform(mess_test)

# Models
nb_model = MultinomialNB()
lr_model = LogisticRegression(C=1, solver='liblinear')
svm_model = SVC(kernel='linear', probability=True)

# Ensemble Model
voting_model = VotingClassifier(
    estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ],
    voting='soft'
)

# Train the models
nb_model.fit(features_train, cat_train)
lr_model.fit(features_train, cat_train)
svm_model.fit(features_train, cat_train)
voting_model.fit(features_train, cat_train)

# Evaluate the Ensemble Model
voting_predictions = voting_model.predict(features_test)
print("Ensemble Model Accuracy:", accuracy_score(cat_test, voting_predictions))
print("Classification Report:\n", classification_report(cat_test, voting_predictions))

# Streamlit Application
st.sidebar.title("Spam Detection System")
st.sidebar.markdown("Navigate through the features using the menu below:")
menu = st.sidebar.radio("Menu", ["Spam Classification", "Data Analysis", "Feedback"])

# Function to predict spam/ham
def predict(message):
    input_message = tfidf.transform([message]).toarray()
    result = voting_model.predict(input_message)
    return result[0]

if menu == "Spam Classification":
    st.header("Classify a Single Message")
    input_message = st.text_input("Enter your message:")
    if st.button("Classify"):
        output = predict(input_message)
        st.write(f"The message is classified as: **{output}**")

    st.header("Batch Processing")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:")
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        uploaded_data['Prediction'] = uploaded_data['text'].apply(predict)
        st.dataframe(uploaded_data)
        st.download_button(
            label="Download Predictions",
            data=uploaded_data.to_csv(index=False),
            file_name='predictions.csv',
            mime='text/csv'
        )

elif menu == "Data Analysis":
    st.header("Dataset Insights")

    # Spam vs Non-Spam Count
    spam_count = data['label'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(spam_count.index, spam_count.values, color=['green', 'red'])
    ax.set_title("Distribution of Spam vs. Non-Spam")
    ax.set_ylabel("Number of Messages")
    ax.set_xlabel("Category")
    st.pyplot(fig)

    # Show a word cloud for spam messages
    st.subheader("Word Cloud for Spam Messages")
    spam_text = " ".join(data[data['label'] == 'Spam']['text'])
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(spam_text)
    st.image(wordcloud.to_array(), caption="Word Cloud for Spam Messages")

elif menu == "Feedback":
    st.header("Provide Feedback")
    feedback_message = st.text_input("Message you think was misclassified:")
    true_label = st.selectbox("What should the correct label be?", ["Not Spam", "Spam"])
    if st.button("Submit Feedback"):
        with open("feedback.csv", "a") as f:
            f.write(f"{feedback_message},{true_label}\n")
        st.write("Thank you for your feedback! This will help improve the model.")
