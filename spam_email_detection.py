import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Larger and diverse dataset of email text and labels (0 for not spam, 1 for spam)
emails = [
    "Get rich Quick! Click here to win a million dollars!",
    "Congratulations! You've been selected for a special promotion.",
    "Claim your prize now! Limited time offer!",
    "Meet singles in your area. Click here now!",
    "URGENT: Your account needs immediate attention!",
    "Exclusive deal for you! Claim your prize today!",
    "You've won a free cruise vacation. Claim now!",
    "Earn $$$$ from home. No investment required!",
    "Unlock the secret to unlimited wealth! Act now!",
    "Special discount on luxury watches and handbags!",
    # Non-Spam Emails
    "Hello, could you please review this document for me",
    "Meeting scheduled for tomorrow, please confirm your attendance.",
    "Dear John, here is the document you requested.",
    "Reminder: Your appointment is tomorrow at 2 PM.",
    "Thank you for your purchase. Your order has been shipped.",
    "Invoice #12345: Payment received. Thank you!",
    "Your account statement for the month of October.",
    "Congratulations on your new job!",
    "Invitation: Annual company dinner on Friday.",
    "Weekly newsletter: Stay updated with our latest products.",
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 

# Data Preprocessing: Convert to lowercase, remove punctuation
emails = [''.join(c for c in email.lower() if c not in string.punctuation) for email in emails]

# Vectorize the emails using TfidfVectorizer with additional configuration
vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=1.0, ngram_range=(1, 2))
x = vectorizer.fit_transform(emails)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print("-----------------------------------------------------")
print(report)
print("-----------------------------------------------------")
# Predict a new email
new_email = ["click here for free money via this link: www"]
new_email = [''.join(c for c in new_email[0].lower() if c not in string.punctuation)]
new_email_vectorized = vectorizer.transform(new_email)
predicted_label = model.predict(new_email_vectorized)

if predicted_label[0] == 0:
    print("\n|| Predicted as NOT SPAM ||\n")
else:
    print("|| Predicted as SPAM ||\n")
