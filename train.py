import os
import re
import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure NLTK resources are downloaded
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------

def generate_synthetic_data(num_samples=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    names = ["John", "Sarah", "Emily", "David", "Michael", "Jessica", "Robert", "Ashley", "Daniel", "Amanda", "James", "Megan"]
    platforms = ["AWS", "Azure", "GitHub", "Jira", "Salesforce", "Stripe", "Docker", "Slack"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    payment_methods = ["Visa", "Mastercard", "Amex", "PayPal"]
    
    billing_templates = [
        ("My credit card was charged twice for the monthly subscription on {date}. Please refund {amount}.", "High"),
        ("I requested a refund of {amount} last week but haven't received it yet.", "High"),
        ("Can I get a copy of my invoice for {month} {year}?", "Low"),
        ("Can you update my billing address on the system? It should be {address}.", "Low"),
        ("I am seeing an unrecognized transaction of {amount} on my card from your site.", "High"),
        ("My payment failed but money was deducted from my bank account.", "High"),
        ("Why was I charged {amount} this month instead of the usual {usual_amount}?", "Medium"),
        ("I want to update my payment method from {pay_method_old} to {pay_method_new}.", "Low"),
        ("Is it possible to switch my billing from monthly to yearly to get a discount?", "Low"),
        ("I received a payment failure notification but my credit card has sufficient funds.", "Medium"),
        ("Can you cancel my subscription and stop auto-renewal? Ticket id: {ticket_id}", "Medium"),
        ("I was billed {amount} during my free trial period. Please check and refund.", "High"),
        ("My billing account is locked and I cannot process the invoice payment.", "High")
    ]
    
    tech_templates = [
        ("The website crashes whenever I try to upload my profile photo or files.", "Medium"),
        ("Database connection timeout error when saving my work on {platform}.", "High"),
        ("The application is extremely slow and takes minutes to load pages.", "High"),
        ("I found a bug: the submit button is greyed out even when forms are complete.", "Medium"),
        ("We are experiencing an API outage and all requests are returning 500 server errors.", "High"),
        ("The search bar returns incorrect results for simple queries.", "Low"),
        ("I get a 404 page not found error when clicking on the dashboard link.", "Medium"),
        ("The mobile app freezes and closes automatically when opening reports.", "Medium"),
        ("My data has not synced between the web app and the mobile application.", "Medium"),
        ("The CSV export feature is broken. It downloads an empty file.", "Medium"),
        ("I am unable to receive password reset emails. The mail server seems down.", "High"),
        ("Our system integration with your webhook is failing with connection refused.", "High"),
        ("I cannot connect to the server. The connection keeps dropping.", "High")
    ]
    
    account_templates = [
        ("I cannot log in to my account. It says password incorrect or user not found.", "Medium"),
        ("Can you reset my password? I forgot it and need access.", "Low"),
        ("I need to delete my account and remove my personal data under GDPR rules.", "Low"),
        ("How do I upgrade my subscription plan to premium?", "Low"),
        ("My account has been suspended and I don't know why. Please reactivate.", "High"),
        ("I want to enable two-factor authentication for my profile for security.", "Low"),
        ("I need to change my registered email from {email_old} to {email_new}.", "Medium"),
        ("How do I add a new team member to my organization profile?", "Low"),
        ("I am unable to change my username. It says username already taken.", "Low"),
        ("My profile information is not updating when I save changes.", "Low"),
        ("Can you merge my two accounts into a single one?", "Low"),
        ("My API key has leaked and I need to regenerate it immediately.", "High"),
        ("Unrecognized login detected on my account. I need to secure it.", "High")
    ]
    
    general_templates = [
        ("What are your business hours and customer support availability on weekends?", "Low"),
        ("Do you offer discounts for non-profit organizations or educational schools?", "Low"),
        ("Where can I find your API documentation and developers SDK guides?", "Low"),
        ("Is there a feature to export reports as PDF or CSV?", "Low"),
        ("I want to request a demo of your enterprise plan for my team.", "Low"),
        ("Can you tell me more about your security and data compliance policies?", "Low"),
        ("Is your service compatible with Internet Explorer 11?", "Low"),
        ("When is the new feature update scheduled for release?", "Low"),
        ("I wanted to share some feedback: the new UI looks really clean!", "Low"),
        ("Do you support integration with Slack or Microsoft Teams?", "Low"),
        ("How can I contact sales for custom licensing terms?", "Low"),
        ("What are the system requirements for running your software?", "Low"),
        ("I would like to suggest a feature: adding a dark mode to the dashboard.", "Low")
    ]
    
    categories = ["Billing", "Technical Issue", "Account Access", "General Query"]
    template_map = {
        "Billing": billing_templates,
        "Technical Issue": tech_templates,
        "Account Access": account_templates,
        "General Query": general_templates
    }
    
    greetings = ["Hello,", "Hi support team,", "Hi,", "Dear Support,", "Hey,", "Hello team,", "Good morning,", "Greetings,"]
    signatures = [
        "\nThanks,\n{name}", 
        "\nBest regards,\n{name}", 
        "\nRegards,\n{name}", 
        "\nThank you,\n{name}",
        "\n- {name}",
        "\nSincerely,\n{name}"
    ]
    
    data = []
    
    for i in range(num_samples):
        # Select category
        category = random.choice(categories)
        # Select template
        template, priority = random.choice(template_map[category])
        
        # Format variables
        date = f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2026"
        amount = f"${random.randint(10, 250)}"
        usual_amount = f"${random.randint(10, 100)}"
        month = random.choice(months)
        year = random.choice(["2025", "2026"])
        address = f"{random.randint(100, 999)} Main St, Cityville"
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        platform = random.choice(platforms)
        name = random.choice(names)
        pay_method_old = random.choice(payment_methods)
        pay_method_new = [p for p in payment_methods if p != pay_method_old][0]
        email_old = f"{name.lower()}{random.randint(10, 99)}@gmail.com"
        email_new = f"{name.lower()}.new@company.com"
        
        # Fill template
        text = template.format(
            date=date, amount=amount, usual_amount=usual_amount, month=month, year=year,
            address=address, ticket_id=ticket_id, platform=platform, pay_method_old=pay_method_old,
            pay_method_new=pay_method_new, email_old=email_old, email_new=email_new
        )
        
        # Add random greetings/signatures with some probability to look natural
        if random.random() > 0.3:
            text = random.choice(greetings) + " " + text
        if random.random() > 0.3:
            text = text + random.choice(signatures).format(name=name)
            
        data.append({
            "ticket_id": f"INC-{2026000 + i}",
            "ticket_text": text,
            "category": category,
            "priority": priority
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(DATA_DIR, "tickets.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} samples at {csv_path}")
    return df

# ---------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove greetings & signatures if they are simple
    # Remove emails and web addresses
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    # Remove special characters and punctuation (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    # Join back to string
    return " ".join(cleaned_tokens)

# ---------------------------------------------------------
# 3. TRAINING & EVALUATION PIPELINE
# ---------------------------------------------------------

def train_and_evaluate():
    # 1. Load data
    df = generate_synthetic_data(num_samples=1000)
    
    print("\nDataset Info:")
    print(df.info())
    print("\nCategory Distribution:\n", df['category'].value_counts())
    print("\nPriority Distribution:\n", df['priority'].value_counts())
    
    # 2. Visualize class distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='category', palette='viridis', order=df['category'].value_counts().index)
    plt.title('Category Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='priority', palette='magma', order=['Low', 'Medium', 'High'])
    plt.title('Priority Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "dataset_distributions.png"), dpi=300)
    plt.close()
    
    # Save individual distributions as requested in implementation plan
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='category', palette='viridis', order=df['category'].value_counts().index)
    plt.title('Support Ticket Categories')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "category_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='priority', palette='magma', order=['Low', 'Medium', 'High'])
    plt.title('Support Ticket Priorities')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "priority_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Preprocessing
    print("Preprocessing ticket text...")
    df['cleaned_text'] = df['ticket_text'].apply(clean_text)
    
    # 4. Feature Engineering
    print("Vectorizing text using TF-IDF...")
    # TF-IDF captures word importance relative to other documents.
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    # Save Vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved vectorizer to {vectorizer_path}")
    
    # Target variables
    y_cat = df['category']
    y_prio = df['priority']
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    }
    
    # ---------------------------------------------------------
    # 5. CATEGORY CLASSIFICATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING CATEGORY CLASSIFICATION MODELS")
    print("="*50)
    
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
    
    best_cat_model = None
    best_cat_score = 0
    best_cat_name = ""
    cat_metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_cat)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test_cat, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_cat, preds, average='weighted')
        
        print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        cat_metrics[name] = {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}
        
        if f1 > best_cat_score:
            best_cat_score = f1
            best_cat_model = model
            best_cat_name = name
            
    print(f"\nBest Category Model: {best_cat_name} with Weighted F1-Score: {best_cat_score:.4f}")
    
    # Save Best Category Model
    cat_model_path = os.path.join(MODELS_DIR, "category_model.pkl")
    joblib.dump(best_cat_model, cat_model_path)
    print(f"Saved category model to {cat_model_path}")
    
    # Evaluate Best Category Model
    best_cat_preds = best_cat_model.predict(X_test)
    print("\nBest Category Model Classification Report:")
    print(classification_report(y_test_cat, best_cat_preds))
    
    # Plot Category Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_cat = confusion_matrix(y_test_cat, best_cat_preds)
    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_cat_model.classes_, 
                yticklabels=best_cat_model.classes_)
    plt.title(f'Category Confusion Matrix ({best_cat_name})')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "category_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 6. PRIORITY PREDICTION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING PRIORITY PREDICTION MODELS")
    print("="*50)
    
    X_train_p, X_test_p, y_train_prio, y_test_prio = train_test_split(X, y_prio, test_size=0.2, random_state=42, stratify=y_prio)
    
    best_prio_model = None
    best_prio_score = 0
    best_prio_name = ""
    prio_metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_p, y_train_prio)
        preds = model.predict(X_test_p)
        
        acc = accuracy_score(y_test_prio, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_prio, preds, average='weighted')
        
        print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        prio_metrics[name] = {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}
        
        if f1 > best_prio_score:
            best_prio_score = f1
            best_prio_model = model
            best_prio_name = name
            
    print(f"\nBest Priority Model: {best_prio_name} with Weighted F1-Score: {best_prio_score:.4f}")
    
    # Save Best Priority Model
    prio_model_path = os.path.join(MODELS_DIR, "priority_model.pkl")
    joblib.dump(best_prio_model, prio_model_path)
    print(f"Saved priority model to {prio_model_path}")
    
    # Evaluate Best Priority Model
    best_prio_preds = best_prio_model.predict(X_test_p)
    print("\nBest Priority Model Classification Report:")
    print(classification_report(y_test_prio, best_prio_preds))
    
    # Plot Priority Confusion Matrix
    priority_order = ['Low', 'Medium', 'High']
    # Adjust matrix based on alphabetical sort order from Scikit-Learn or enforce explicit classes
    classes = list(best_prio_model.classes_)
    
    plt.figure(figsize=(8, 6))
    cm_prio = confusion_matrix(y_test_prio, best_prio_preds, labels=classes)
    sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title(f'Priority Confusion Matrix ({best_prio_name})')
    plt.ylabel('True Priority')
    plt.xlabel('Predicted Priority')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "priority_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Save a JSON file detailing execution results
    import json
    results = {
        "category_results": {
            "best_model": best_cat_name,
            "metrics": cat_metrics,
            "report": classification_report(y_test_cat, best_cat_preds, output_dict=True)
        },
        "priority_results": {
            "best_model": best_prio_name,
            "metrics": prio_metrics,
            "report": classification_report(y_test_prio, best_prio_preds, output_dict=True)
        }
    }
    results_path = os.path.join(MODELS_DIR, "metrics_report.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation metrics to {results_path}")

if __name__ == "__main__":
    train_and_evaluate()
