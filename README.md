# 🎓 Evergreen School Chatbot

This is a smart, intent-based chatbot designed to assist users with school-related inquiries such as admissions, fees, event hall bookings, feedback, and more. It uses a combination of keyword matching, synonym expansion, and machine learning to understand user input and respond appropriately.

---

## 🚀 Project Overview

The chatbot is built using Python and leverages libraries like NLTK, TextBlob, spaCy, and scikit-learn. It processes user input, predicts intent, and generates context-aware responses using curated datasets stored in CSV files.

---

## 🧠 Key Features

- **Intent Classification** using Naive Bayes
- **Synonym Expansion** for flexible language understanding
- **Spelling Correction** with TextBlob
- **POS Tagging** to extract meaningful content
- **Sentiment Analysis** for tone-aware replies
- **Profanity Detection** for respectful conversations
- **Custom Handlers** for each intent category
- **CSV-Based Response Bank** for scalable content

---

## 📂 Folder Structure
EvergreenChatbot/
├── chatbot.py
├── .venv/
├── CHAT BOT ADMISSIONS DATA.csv
├── CHAT BOT FEES DATA.csv
├── CHAT BOT MULTIPURPOSE HALL DATA.csv
├── CHAT BOT GENERAL INFORMATION DATA.csv
├── CHAT BOT CONTACT INFORMATION DATA.csv
├── CHAT BOT CAREER DATA.csv
├── CHAT BOT BUSINESS PARTNERSHIP DATA.csv
├── CHAT BOT FEEDBACK DATA.csv
├── CHAT BOT GREETINGS DATA.csv
├── CHAT BOT PROFANITY DATA.csv
├── CHAT BOT PHYSICAL APPOINTMENT DATA.csv

---

## 🧩 Intent Categories

| Intent                | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `admissions`         | School entry, registration, and application queries          |
| `fees`               | Tuition, payment methods, discounts, and financial info      |
| `physical_appointment` | Medical checkups and admission requirements                 |
| `multi_purpose_hall` | Booking and usage of school event spaces                     |
| `general_information`| School mission, values, and educational philosophy           |
| `contact_information`| Phone, email, and location details                           |
| `careers`            | Job opportunities and recruitment                            |
| `business_partnership`| Sponsorships and strategic collaborations                   |
| `feedback`           | Suggestions, complaints, and reviews                         |
| `greetings`          | Polite greetings and conversational openers                  |
| `profanity`          | Inappropriate language detection and moderation              |

---

## 🧠 How It Works

1. **User Input** is cleaned, tokenized, and corrected.
2. **Synonym Matching** checks for keywords and their variants.
3. **Intent Prediction** uses both rule-based and ML-based scoring.
4. **Response Generation** pulls from CSV datasets or custom handlers.
5. **Fallback Logic** handles unknown inputs with sentiment-aware replies.
6. **Exit Detection** gracefully ends the chat when the user is done.

---

## 🛠 Technologies Used

- `NLTK` for tokenization, POS tagging, and classification
- `TextBlob` for spelling correction and sentiment analysis
- `spaCy` for entity recognition
- `scikit-learn` for vectorization and similarity scoring
- `pandas` for CSV data handling

---

## 📈 Future Improvements

- Add GUI or web interface
- Expand training data for better accuracy
- Integrate with messaging platforms
- Add multilingual support

---

## 🙌 Author

Built by **Isedehi Aigbogun** with care and curiosity.  
This chatbot is designed to make school inquiries smoother, smarter, and more human.
