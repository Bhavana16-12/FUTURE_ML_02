FUTURE_ML_02
# Support Ticket Classification and Prioritization

## Project Overview

This project is developed to automate the process of handling customer support tickets. In many organizations, support teams spend a lot of time reading tickets, identifying the issue, and deciding how urgent it is. This system helps reduce that effort by using Machine Learning and Natural Language Processing (NLP).

The model predicts:

* The category of the ticket
* The priority level of the ticket

The project also includes a Streamlit web application where users can enter a support ticket and get predictions instantly.

---

## Categories Supported

The system can classify tickets into the following categories:

* Billing
* Technical Issue
* Account Access
* General Query

---

## Priority Levels

The model predicts one of the following priority levels:

* High
* Medium
* Low

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Joblib
* Matplotlib
* Seaborn
* Streamlit

---

## Project Structure

```text
Support-Ticket-Classification/
│
├── data/
│   └── tickets.csv
├── notebooks/
│   └── ticket_classifier.ipynb
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── vectorizer.pkl
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

After running the above command, open the local URL shown in the terminal.

---

## Machine Learning Process

The following steps were used while building the model:

1. Text cleaning and preprocessing
2. Tokenization and stopword removal
3. TF-IDF feature extraction
4. Model training and evaluation
5. Saving the best model for predictions

Different machine learning algorithms were tested and the best-performing model was selected.

---

## Features

* Automatic ticket classification
* Priority prediction
* Easy-to-use web interface
* Fast predictions
* Helps reduce manual effort in support teams

---

## Future Improvements

* Support for more ticket categories
* Multi-language support
* Integration with helpdesk platforms
* Deep learning-based models for better accuracy

---

## Conclusion

This project demonstrates how Machine Learning and NLP can be used to automate support ticket management. It helps classify customer issues and identify their urgency, making the support process faster and more efficient.
