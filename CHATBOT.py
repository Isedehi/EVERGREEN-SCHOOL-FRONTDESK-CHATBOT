# School Chatbot with Improved Intent Matching

#Importing necessary libraries 
import nltk # pip install nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics import edit_distance
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk import pos_tag, word_tokenize, bigrams
import pandas as pd  #pip install pandas
from textblob import TextBlob  # pip install textblob ,  python -m textblob.download_corpora
from sklearn.feature_extraction.text import CountVectorizer    #pibpp install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB   # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # OR Decision Tree
from sklearn.linear_model import LogisticRegression     # OR Logistic Regression


# Downloading required NLTK data packages for tokenization, sentiment analysis, and part-of-speech tagging
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Welcome message
print("Evergreen Admin: Welcome to Evergreen School Front-Desk Chatbot")
print("Evergreen Admin: How can I assist you today? Are you here for admissions, to find out about our multi-purpose hall, or to provide feedback?")

# Initializing core NLP components:
# - Porter Stemmer for word stemming
# - English stopword list for filtering
# - VADER Sentiment Analyzer for tone detection
# - WordNet Lemmatizer for word normalization
# - spaCy's small English model for advanced NLP tasks (e.g., entity recognition)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
classifier = None

# Preparing sample training data for a simple text classification task:
# - X_train contains example user inputs (greetings and farewells)
# - y_train labels each input with its corresponding intent
X_train = ["Hi", "Hello", "Goodbye", "See you later"]
y_train = ["greeting", "greeting", "farewell", "farewell"]

# Vectorizing text using TF-IDF to convert raw strings into numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Normalizing and vectorizing a sample user input for prediction

user_input = "Hi there!"
normalized_input = user_input.lower()
user_vec = vectorizer.transform([user_input])

models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

#Implementing Pos tag 
text = "My child is going to primary 3 next term."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Defining intents and responses for the chatbot, covering various topics such as admissions, fees, and general information
intents = {
    'admissions': 'We\'re excited you\'re considering joining us! Admissions are typically accepted at the start of each term, with limited exceptions. Please contact our office to discuss the best time for your child\'s admission, as mid-term admissions may not be feasible and third-term admissions are generally not advised.',
    'fees': 'Our fees cover tuition, books/stationery, uniform pack, utilities, exams and records, maintenance/development, and PTA levy. Events are complimentary for the first term only. Payments can be made via bank transfer to our listed accounts. No cash payments are accepted. We\'d be happy to provide a detailed breakdown of costs if needed.',
    'physical_appointment': 'For children above 2 years old, a physical examination is required before admission. Please contact our office to schedule an appointment. Children aged 2 and below are exempt and will be placed in our beginner\'s class.',
    'multi_purpose_hall': 'Our event hall is available for booking! Reach out to our events coordinator to check availability and learn about the booking process.',
    'general_information': 'Our mission is to provide quality education and foster growth in a supportive environment. We value curiosity, creativity, and community.',
    'contact_information': 'You can reach us at 08139060571 or admin@emailevergreen.sch.ng. Our school is located at 20/22, Majaro street, Onike-Yaba, Lagos. We’d love to hear from you!',
    'careers': 'We are always looking for passionate educators to join our team! If you are interested in exploring job opportunities, you can submit your application by emailing your resume and a brief introduction, or preferably by visiting our school in person to submit a physical copy. We prioritize local candidates, as we believe it fosters a better work environment and community connection. We look forward to reviewing your application and discussing how you can contribute to our school community!',
    'business_partnership': 'We welcome business partnerships and sponsorships. Please contact us to discuss potential collaboration opportunities and how we can work together to support our school community.',
    'feedback': 'We value your feedback and suggestions. Please share your thoughts with us, and we will do our best to address any concerns or improve our services. Your input is important to us!',
    'greetings': 'Welcome to Evergreen School Chatbot! How can I assist you today? Are you here for admissions, to find out about our multi-purpose hall or to provide feedback. I am here to assist you.',
    'profanity': "Oof! That’s a bold choice of words. Want to vent or shall we keep it classy?"
}
# Defining keywords for each intent to facilitate accurate matching and response generation
keywords = {

'admissions': ['admission', 'apply', 'enroll', 'placement', 'age', 'requirement', 'term', 'child', 'son', 'daughter',
'register', 'mid-term', 'documents', 'start date', 'duration', 'lengthy', 'class', 'administrator', 'admin', 'admin officer', 'admin office',
'tour', 'criteria', 'specific requirements', 'submit', 'deadline',
'online application', 'portal', 'track status', 'application process', 'KG', 'kindergarten','nursery', 'primary', 'basic', 'year',
'notification', 'appeal', 'special requests', 'special needs', 'accommodate', '1', '2', '2.5', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],

'fees': ['tuition', 'bank transfer', 'PTA levy', 'payment methods',
'school fees', 'hidden charges', 'book fees', 'discounts',
'new intake fees', 'payment process', 'online payment',
'missed payment', 'refundable fees', 'payment receipt',
'advance payment', 'maintenance fee', 'utilities fee',
'fee breakdown', 'exam fee', 'uniform cost', 'payment plan',
'late payment', 'scholarship', 'fee waiver', 'additional fees',
'credit card payment', 'invoice', 'applicable fees',
'mobile banking', 'payment deadline', 'fee refund',
'fee statement', 'extracurricular fees', 'siblings discount',
'lost receipt', 'direct debit'],

'physical_appointment': ['examination', 'assessment', 'schedule', 'checkup', 'medical',
'requirement', '2 years old', 'appointment', 'phone',
'prepare', 'results', 'specific doctor', 'frequency',
'sick', 'report', 'eligibility', 'unsure', 'online scheduling',
'cancel', 'reschedule', 'medical condition', 'on-site',
'validity', 'questions', 'impact', 'documentation','administrator', 'admin', 'admin officer', 'admin office',
'notification', 'follow-up'],

'multi_purpose_hall': ['hall', 'booking', 'facility', 'venue', 'stage',
'backstage', 'rent', 'availability', 'capacity',
'guest', 'seating', 'catering', 'audio-visual',
'decorations', 'decoratewhat about the price of the hall' 'customization', 'cancellation',
'refund', 'events coordinator', 'tour', 'rules',
'regulations', 'quote', 'discounts', 'discount', 'finalize',
'booking process', 'occasion', 'party'],

'general_information': ['mission', 'values', 'school', 'education', 'curiosity', 'creativity', 'community', 
'curriculum', 'extracurricular', 'sports', 'character development', 'arts', 'music', 
'interests', 'focus', 'clubs', 'organizations', 'relationships', 'staff', 'approach', 
'core values', 'community service', 'academic growth', 'unique', 'environment', 
'parents', 'parent', 'involvement', 'integrity', 'respect', 'sports teams', 'well-rounded', 
'excellence', 'support system', 'nurture', 'philosophy', 'vision', 'goals', 'administrator', 'admin', 'admin officer', 'admin office',
'achievement', 'success', 'student life', 'campus', 'facilities', 'resources', 'founded',
'opportunities', 'expectations', 'teachers', 'teacher', 'students', 'student', 'learning environment', 'founder'],

'contact_information': ['contact', 'contact details','phone', 'email', 'address', 'location', 'located',
'08139060571', 'admin@evergreen.sch.ng', 'number',
'reach', 'call', 'mobile', 'contact number',
'best number', 'via phone', 'email address',
'email you', 'via email', 'official email',
'physical address', 'located', 'get address','administrator', 'admin', 'admin officer', 'admin office',
'office address', 'find you', 'provide location', 'locate the school',
'get to location', 'visit', 'locate','school location', 'school address', 'reach the admission office', 'reach the school',],

'careers': ['job', 'vacancy', 'employment', 'position',
'career', 'opportunity', 'recruitment',
'openings', 'application process', 'resume',
'job openings', 'best way to apply', 'careers page',
'submit application', 'hiring', 'email resume',
'send application', 'current openings',
'find job vacancies', 'application online',
'qualified', 'job postings', 'future openings',
'stay updated', 'multiple positions',
'application form', 'application duration',
'internship', 'follow up', 'feedback'],


'business_partnership': ['partnership', 'proposal', 'product', 'service',
'collaboration', 'business', 'sponsorship',
'donation', 'company', 'meeting',
'strategic plans', 'priorities', 'challenges',
'solutions', 'success stories', 'vendor relationships',
'long-term', 'expertise', 'goals',
'resources', 'achieve goals', 'high-quality',
'tailor solutions', 'evaluation', 'vision',
'pilot programs', 'trials', 'decision factors',
'additional information', 'follow-up'],

'feedback': ['feedback', 'suggestion', 'complaint', 'review',
'opinion', 'comment', 'provide feedback',
'share feedback', 'give feedback', 'handle feedback',
'value feedback', 'share thoughts',
'report issue', 'complain', 'filing a complaint',
'resolve complaints', 'appreciate reviews',
'leave a review', 'suggest improvements',
'use feedback', 'report problem',
'take feedback seriously', 'make a suggestion'], 

'greetings': ['hi', 'hello', 'good morning', 'good afternoon',
'good evening', 'hey', 'hey there', 'hello there', 'hi there', 'help', 'assist', 
'information', 'answer', 'questions', 'need help', 'can you help', 'assist me', 
'provide information', 'i need assistance', 'thanks', 'thank you', 'appreciate', 'grateful', 
'thankful', 'bye', 'goodbye', 'take care', 'see you later', 'have a great day'], 

'profanity': ['fuck', 'f***', 'bullshit', 'bullsh**', 'shit', 'damn', 'bastard',
'f*** you', 'fuck you', 'f*** off', 'fuck off', 'piece of shit', 'piece of s***',
'this is bullshit', 'this is bullsh**', 'f*** your rules', 'fuck your rules',
'f*** your advice', 'fuck your advice', 'f*** this chatbot', 'fuck this chatbot',
'you are useless', 'you’re f***ing useless', 'you’re useless', 'you suck',
'this sucks', 'this is stupid', 'this is dumb', 'you’re dumb', 'you’re stupid',
'shut up', 'i hate this', 'i hate this chatbot', 'you’re annoying', 'so annoying',
'this is a nightmare', 'this is a f***ing nightmare', 'this is ridiculous',
'this is f***ing ridiculous', 'you’re a joke', 'you’re a failure',
'you’re incompetent', 'you’re a piece of s***', 'this is a waste of time',
'this is a waste of my time', 'i’m done with this', 'i’m fed up', 'i’m outta here',
'this is boring', 'this is f***ing boring', 'you’re not helpful', 'you’re not smart',
'you’re not funny', 'you’re not listening', 'you’re a waste of space',
'i’m disgusted', 'i’m sick of this', 'f*** this', 'f*** this s***']

}

# Defining Custom Synonyms for the Keyword (intended to override WordNet's base

keyword_synonyms = {
# Keyword, admission synonyms
'admission': ['entrance', 'access', 'entry', 'acceptance', 'enrollment', 'registration'],
'apply': ['request', 'submit', 'petition'],
'enroll': ['register', 'sign up', 'matriculate'],
'placement': ['positioning', 'location', 'allocation'],
'age': ['maturity', 'seniority', 'longevity'],
'requirement': ['necessity', 'prerequisite', 'obligation'],
'term': ['period', 'duration', 'semester'],
'register': ['sign up', 'enroll', 'record'],
'mid-term': ['halfway', 'interim', 'midpoint'],
'documents': ['papers', 'records', 'files'],
'start date': ['commencement', 'beginning', 'initiation'],
'duration': ['length', 'period', 'span'],
'lengthy': ['prolonged', 'extended', 'protracted'],
'tour': ['excursion', 'trip', 'visit'],
'criteria': ['standards', 'benchmarks', 'guidelines'],
'specific requirements': ['particular needs', 'precise specifications', 'detailed criteria'],
'submit': ['yield', 'surrender', 'present'],
'deadline': ['cutoff', 'limit', 'time limit'],
'online application': ['digital form', 'web application', 'electronic submission'],
'portal': ['gateway', 'entrance', 'interface'],
'track status': ['monitor progress', 'follow up', 'check status'],
'application process': ['procedure', 'protocol', 'sequence'],
'notification': ['alert', 'message', 'announcement'],
'appeal': ['request', 'petition', 'plea'],
'special requests': ['particular needs', 'specific asks', 'unique requirements'],
'special needs': ['particular requirements', 'specific necessities', 'unique needs'],
'accommodate': ['adapt', 'adjust', 'provide for'],

# Keyword, fees synonyms
'fees': ['charges', 'rates', 'tariffs', 'how much', 'tuition amount','price', 'registration fee', 'admission fee'],
'tuition': ['instructional fees', 'course fees', 'education costs'],
'bank transfer': ['wire transfer', 'electronic payment', 'funds transfer'],
'PTA levy': ['parent-teacher association fee', 'school levy', 'educational tax'],
'payment methods': ['payment options', 'payment modes', 'payment channels'],
'school fees': ['tuition fees', 'education fees', 'school charges'],
'hidden charges': ['additional fees', 'extra charges', 'surcharge'],
'book fees': ['book costs', 'textbook fees', 'study material fees'],
'discounts': ['reductions', 'concessions', 'price cuts'],
'new intake fees': ['admission fees', 'enrollment fees', 'registration fees'],
'payment process': ['payment procedure', 'payment protocol', 'payment system'],
'online payment': ['digital payment', 'electronic payment', 'web payment'],
'missed payment': ['overdue payment', 'late payment', 'unpaid bill'],
'refundable fees': ['reimbursable fees', 'returnable fees', 'refundable charges'],
'payment receipt': ['payment confirmation', 'payment proof', 'receipt'],
'advance payment': ['prepayment', 'upfront payment', 'deposit'],
'maintenance fee': ['upkeep fee', 'maintenance charges', 'service fee'],
'utilities fee': ['utility charges', 'service charges', 'overhead costs'],
'fee breakdown': ['fee structure', 'fee details', 'fee schedule'],
'exam fee': ['examination fee', 'test fee', 'assessment fee'],
'uniform cost': ['uniform fee', 'uniform price', 'school uniform cost'],
'payment plan': ['payment schedule', 'payment arrangement', 'installment plan'],
'late payment': ['overdue payment', 'delayed payment', 'missed payment'],
'scholarship': ['grant', 'bursary', 'educational award'],
'fee waiver': ['fee exemption', 'fee reduction', 'fee discount'],
'additional fees': ['extra charges', 'supplementary fees', 'additional costs'],
'credit card payment': ['card payment', 'credit card transaction', 'online payment'],
'invoice': ['bill', 'statement', 'payment request'],
'applicable fees': ['relevant fees', 'applicable charges', 'required fees'],
'mobile banking': ['mobile payment', 'mobile transaction', 'mobile money transfer'],
'payment deadline': ['payment due date', 'payment cutoff', 'payment timeframe'],
'fee refund': ['fee reimbursement', 'fee return', 'refund'],
'fee statement': ['fee invoice', 'fee bill', 'fee breakdown'],
'extracurricular fees': ['activity fees', 'club fees', 'sports fees'],
'siblings discount': ['family discount', 'sibling reduction', 'family rate'],
'lost receipt': ['missing receipt', 'lost payment proof', 'unavailable receipt'],
'direct debit': ['automatic payment', 'direct payment', 'auto-debit'],


# Keyword, physical_appointment synonyms
'physical_appointment': ['appointment', 'doctor/s visit', 'checkup'],
'examination': ['checkup', 'assessment', 'evaluation'],
'assessment': ['evaluation', 'appraisal', 'examination'],
'schedule': ['timetable', 'appointment book', 'diary'],
'checkup': ['medical checkup', 'health checkup', 'physical examination'],
'medical': ['healthcare', 'clinical', 'health'],
'requirement': ['necessity', 'prerequisite', 'obligation'],
'2 years old': ['toddler', 'young child', 'preschooler'],
'appointment': ['meeting', 'consultation', 'session'],
'phone': ['telephone', 'mobile phone', 'cell phone'],
'prepare': ['get ready', 'prepare oneself', 'make preparations'],
'results': ['outcome', 'findings', 'conclusion'],
'specific doctor': ['particular doctor', 'chosen doctor', 'designated doctor'],
'frequency': ['recurrence', 'regularity', 'interval'],
'sick': ['ill', 'unwell', 'ailing'],
'report': ['document', 'record', 'summary'],
'eligibility': ['qualification', 'suitability', 'entitlement'],
'unsure': ['uncertain', 'doubtful', 'unclear'],
'online scheduling': ['digital scheduling', 'electronic scheduling', 'web scheduling'],
'cancel': ['terminate', 'abort', 'call off'],
'reschedule': ['rearrange', 'resetime', 'replan'],
'medical condition': ['health issue', 'medical problem', 'health condition'],
'on-site': ['in-person', 'face-to-face', 'physical presence'],
'validity': ['legitimacy', 'authenticity', 'accuracy'],
'questions': ['inquiries', 'queries', 'interrogations'],
'impact': ['effect', 'influence', 'consequence'],
'documentation': ['records', 'documents', 'papers'],
'notification': ['alert', 'message', 'announcement'],
'follow-up': ['check-in', 'review', 'monitoring'],


# Keyword, multi_purpose_hall synonyms

'multi_purpose_hall': ['event space', 'function hall', 'hall'],
'hall': ['auditorium', 'theater', 'conference room'],
'booking': ['reservation', 'appointment', 'scheduling'],
'facility': ['venue', 'space', 'infrastructure'],
'venue': ['location', 'site', 'event space'],
'stage': ['platform', 'podium', 'performance area'],
'backstage': ['behind-the-scenes', 'dressing room', 'wings'],
'rent': ['lease', 'hire', 'charter'],
'availability': ['schedule', 'openings', 'bookings'],
'capacity': ['seating capacity', 'room for', 'accommodate'],
'guest': ['visitor', 'attendee', 'participant'],
'seating': ['chairs', 'tables', 'accommodations'],
'catering': ['food service', 'hospitality', 'refreshments'],
'audio-visual': ['AV equipment', 'sound and lighting', 'multimedia'],
'decorations': ['décor', 'furnishings', 'ambiance'],
'customization': ['personalization', 'tailoring', 'bespoke'],
'cancellation': ['termination', 'annulment', 'call off'],
'refund': ['reimbursement', 'repayment', 'return'],
'events coordinator': ['event planner', 'wedding planner', 'conference coordinator'],
'tour': ['guided tour', 'inspection', 'viewing'],
'rules': ['regulations', 'guidelines', 'policies'],
'regulations': ['rules', 'laws', 'ordinances'],
'quote': ['estimate', 'bid', 'proposal'],
'discounts': ['reductions', 'concessions', 'price cuts'],
'discount': ['reduction', 'concession', 'price cut'],
'finalize': ['confirm', 'settle', 'complete'],
'booking process': ['reservation process', 'scheduling process', 'booking procedure'],
'occasion': ['event', 'celebration', 'gathering'],
'party': ['gathering', 'celebration', 'social event'],


# Keyword, general_information synonyms
'general_information': ['basic facts', 'essential details', 'overview'],
'mission': ['purpose', 'aim', 'objective'],
'values': ['principles', 'ethics', 'standards'],
'school': ['institution', 'academy', 'educational establishment'],
'education': ['learning', 'instruction', 'training'],
'curiosity': ['inquisitiveness', 'interest', 'inquiring mind'],
'creativity': ['imagination', 'innovation', 'originality'],
'community': ['society', 'neighborhood', 'collective'],
'curriculum': ['syllabus', 'course of study', 'educational program'],
'extracurricular': ['non-academic', 'outside class', 'supplementary'],
'sports': ['athletics', 'games', 'physical activity'],
'character development': ['personal growth', 'moral development', 'self-improvement'],
'arts': ['creative arts', 'fine arts', 'performing arts'],
'music': ['musical education', 'music program', 'artistic expression'],
'interests': ['hobbies', 'passions', 'enthusiasms'],
'focus': ['concentration', 'emphasis', 'priority'],
'clubs': ['organizations', 'groups', 'societies'],
'organizations': ['associations', 'groups', 'committees'],
'relationships': ['connections', 'bonds', 'interactions'],
'staff': ['faculty', 'teachers', 'personnel'],
'approach': ['method', 'technique', 'strategy'],
'core values': ['fundamental principles', 'key values', 'essential beliefs'],
'community service': ['volunteer work', 'service projects', 'social responsibility'],
'academic growth': ['intellectual development', 'educational progress', 'knowledge acquisition'],
'unique': ['distinctive', 'special', 'exceptional'],
'environment': ['setting', 'atmosphere', 'surroundings'],
'parents': ['guardians', 'families', 'caregivers'],
'involvement': ['participation', 'engagement', 'commitment'],
'integrity': ['honesty', 'trustworthiness', 'ethics'],
'respect': ['esteem', 'regard', 'consideration'],
'sports teams': ['athletic teams', 'sports squads', 'competitive teams'],
'well-rounded': ['balanced', 'diverse', 'multifaceted'],
'excellence': ['outstanding quality', 'superior performance', 'distinction'],
'support system': ['network', 'resources', 'assistance'],
'nurture': ['care', 'development', 'cultivation'],
'philosophy': ['ideology', 'principles', 'beliefs'],
'vision': ['mission', 'goal', 'aspiration'],
'goals': ['objectives', 'targets', 'aims'],
'achievement': ['accomplishment', 'success', 'attainment'],
'success': ['achievement', 'accomplishment', 'triumph'],
'student life': ['campus life', 'school experience', 'academic life'],
'campus': ['school grounds', 'university premises', 'facilities'],
'facilities': ['resources', 'equipment', 'infrastructure'],
'resources': ['assets', 'materials', 'support'],
'opportunities': ['chances', 'prospects', 'options'],
'expectations': ['anticipations', 'hopes', 'requirements'],
'founder': ['creator', 'establisher', 'originator', 'owner', 'proprietor', 'proprietress', 'school owner'],


# Keyword, contact_information synonyms

'contact_information': ['contact details', 'reach us', 'get in touch'],
'contact': ['get in touch', 'reach out', 'communicate'],
'phone': ['telephone', 'mobile phone', 'cell phone'],
'email': ['electronic mail', 'e-mail', 'digital message'],
'address': ['location', 'physical address', 'postal address'],
'location': ['whereabouts', 'site', 'venue', 'located'],
'number': ['phone number', 'contact number', 'digits'],
'reach': ['contact', 'get in touch', 'communicate with'],
'call': ['phone', 'ring', 'dial'],
'mobile': ['cell phone', 'smartphone', 'mobile device'],
'contact number': ['phone number', 'contact info', 'reach us'],
'best number': ['preferred contact', 'best way to reach', 'primary number'],
'via phone': ['by phone', 'over the phone', 'through phone'],
'email address': ['e-mail address', 'digital address', 'online contact'],
'email you': ['send you an email', 'contact you via email', 'reach out via email'],
'via email': ['by email', 'through email', 'electronically'],
'official email': ['work email', 'professional email', 'business email'],
'physical address': ['mailing address', 'street address', 'location', 'located'],
'located': ['situated', 'based', 'found'],
'get address': ['find location', 'obtain address', 'get directions'],
'office address': ['work address', 'business address', 'company location', 'located', 'location'],
'find you': ['locate you', 'find your location', 'get directions to'],
'provide location': ['give directions', 'share location', 'provide address', 'located'],
'get to location': ['find your way', 'reach your destination', 'arrive at'],
'visit': ['stop by', 'drop in', 'come see'],


# Keyword, careers synonyms

'careers': ['job opportunities', 'employment options', 'career paths'],
'job': ['position', 'role', 'employment opportunity'],
'vacancy': ['opening', 'available position', 'job opening'],
'employment': ['work', 'job', 'occupation'],
'position': ['job', 'role', 'post'],
'career': ['profession', 'occupation', 'vocation'],
'opportunity': ['chance', 'possibility', 'opening'],
'recruitment': ['hiring', 'staffing', 'talent acquisition'],
'openings': ['job openings', 'available positions', 'vacancies'],
'application process': ['job application', 'application procedure', 'hiring process'],
'resume': ['CV', 'curriculum vitae', 'job application document'],
'job openings': ['vacancies', 'available positions', 'job opportunities'],
'best way to apply': ['application method', 'submission process', 'preferred application channel'],
'careers page': ['job board', 'career section', 'employment page'],
'submit application': ['apply', 'send application', 'submit job application'],
'hiring': ['recruitment', 'staffing', 'employment'],
'email resume': ['send CV', 'submit resume via email', 'email job application'],
'send application': ['submit job application', 'apply for job', 'send resume'],
'current openings': ['available positions', 'current job openings', 'current vacancies'],
'find job vacancies': ['search for jobs', 'look for job openings', 'find employment opportunities'],
'application online': ['online application', 'digital application', 'web-based application'],
'qualified': ['eligible', 'suitable', 'experienced'],
'job postings': ['job ads', 'job listings', 'employment opportunities'],
'future openings': ['upcoming jobs', 'future job opportunities', 'anticipated vacancies'],
'stay updated': ['keep informed', 'follow updates', 'track news'],
'multiple positions': ['several jobs', 'multiple job openings', 'various positions'],
'application form': ['job application form', 'employment application', 'application template'],
'application duration': ['application period', 'application deadline', 'submission timeframe'],
'internship': ['work experience', 'traineeship', 'intern position'],
'follow up': ['check in', 'follow up on application', 'inquire about status'],
'feedback': ['response', 'comment', 'evaluation'],


# Keyword, business_partnership synonyms

'business_partnership': ['strategic alliance', 'business collaboration', 'joint venture'],
'partnership': ['collaboration', 'alliance', 'joint effort'],
'proposal': ['offer', 'suggestion', 'pitch'],
'product': ['goods', 'commodity', 'service'],
'service': ['support', 'assistance', 'solution'],
'collaboration': ['partnership', 'cooperation', 'teamwork'],
'business': ['enterprise', 'company', 'organization'],
'sponsorship': ['support', 'patronage', 'endorsement'],
'donation': ['contribution', 'gift', 'grant'],
'company': ['firm', 'enterprise', 'organization'],
'meeting': ['gathering', 'conference', 'discussion'],
'strategic plans': ['business strategy', 'long-term plans', 'goals'],
'priorities': ['key objectives', 'main focus', 'top priorities'],
'challenges': ['obstacles', 'issues', 'problems'],
'solutions': ['answers', 'resolutions', 'fixes'],
'success stories': ['case studies', 'testimonials', 'achievements'],
'vendor relationships': ['supplier relationships', 'business partnerships', 'collaborations'],
'long-term': ['sustainable', 'enduring', 'lasting'],
'expertise': ['specialized knowledge', 'professional skills', 'expert advice'],
'goals': ['objectives', 'targets', 'aims'],
'resources': ['assets', 'materials', 'support'],
'achieve goals': ['meet objectives', 'reach targets', 'accomplish aims'],
'high-quality': ['excellent', 'superior', 'top-notch'],
'tailor solutions': ['customize solutions', 'personalize services', 'adapt to needs'],
'evaluation': ['assessment', 'review', 'analysis'],
'vision': ['mission', 'goal', 'aspiration'],
'pilot programs': ['test projects', 'trial initiatives', 'experimental programs'],
'trials': ['tests', 'experiments', 'pilot studies'],
'decision factors': ['key considerations', 'important factors', 'decision criteria'],
'additional information': ['further details', 'more information', 'supplementary data'],
'follow-up': ['check-in', 'update', 'progress review'],

# Keyword, feedback synonyms

'feedback': ['input', 'response', 'reaction'],
'suggestion': ['recommendation', 'idea', 'proposal'],
'complaint': ['grievance', 'concern', 'objection'],
'review': ['evaluation', 'assessment', 'rating'],
'opinion': ['viewpoint', 'perspective', 'thoughts'],
'comment': ['remark', 'statement', 'observation'],
'provide feedback': ['give input', 'offer feedback', 'share thoughts'],
'share feedback': ['provide feedback', 'give feedback', 'share opinions'],
'give feedback': ['offer feedback', 'provide input', 'share thoughts'],
'handle feedback': ['manage feedback', 'process feedback', 'respond to feedback'],
'value feedback': ['appreciate feedback', 'cherish feedback', 'prioritize feedback'],
'share thoughts': ['express opinions', 'provide input', 'give feedback'],
'report issue': ['report problem', 'notify issue', 'flag concern'],
'complain': ['express dissatisfaction', 'lodge complaint', 'voice concern'],
'filing a complaint': ['submitting complaint', 'lodging grievance', 'reporting issue'],
'resolve complaints': ['address concerns', 'resolve issues', 'handle grievances'],
'appreciate reviews': ['value feedback', 'cherish reviews', 'thank reviewers'],
'leave a review': ['provide feedback', 'submit review', 'share opinion'],
'suggest improvements': ['offer suggestions', 'propose enhancements', 'recommend changes'],
'use feedback': ['apply feedback', 'incorporate feedback', 'utilize input'],
'report problem': ['notify issue', 'flag concern', 'report concern'],
'take feedback seriously': ['value feedback', 'prioritize feedback', 'consider input'],
'suggest improvements': ['propose changes', 'recommend enhancements', 'offer ideas'],
'make a suggestion': ['offer a suggestion', 'propose an idea', 'recommend a change'],

# Keyword, greetings synonyms

'greetings': ['hello', 'hey', 'greetings to you'],
'hello': ['hi', 'hey', 'hello there'],
'good morning': ['morning', 'good day', 'top of the morning'],
'good afternoon': ['afternoon', 'good day', 'hello'],
'good evening': ['evening', 'good night', 'hello'],
'hey': ['hi', 'hello', 'yo'],
'hey there': ['hi there', 'hello there', 'greetings'],
'hello there': ['hi there', 'hello', 'greetings'],
'hi there': ['hello there', 'hi', 'hey'],
'help': ['assist', 'support', 'aid'],
'assist': ['help', 'support', 'facilitate'],
'information': ['data', 'knowledge', 'facts'],
'answer': ['response', 'reply', 'solution'],
'questions': ['queries', 'inquiries', 'requests'],
'need help': ['require assistance', 'need support', 'need aid'],
'can you help': ['can you assist', 'can you support', 'can you aid'],
'assist me': ['help me', 'support me', 'aid me'],
'provide information': ['give information', 'share knowledge', 'provide data'],
'i need assistance': ['i need help', 'i need support', 'i require aid'],
'thanks': ['thank you', 'appreciation', 'gratitude'],
'thank you': ['thanks', 'appreciation', 'gratitude'],
'appreciate': ['value', 'cherish', 'thank'],
'grateful': ['thankful', 'appreciative', 'obliged'],
'thankful': ['grateful', 'appreciative', 'relieved'],
'bye': ['goodbye', 'farewell', 'see you later'],
'goodbye': ['bye', 'farewell', 'adieu'],
'take care': ['be careful', 'stay safe', 'look after yourself'],
'see you later': ['see you soon', 'until later', 'farewell'],
'have a great day': ['have a nice day', 'enjoy your day', 'all the best'], 

# Profanity synonyms

'fuck': ['f***', 'fuk', 'f@#k', 'f*ck'],
'bullshit': ['bullsh**', 'bs', 'bull sh*t', 'bullsh!t'],
'shit': ['s***', 'sh*t', 'sh!t'],
'damn': ['dam', 'd@mn', 'daamn'],
'bastard': ['b@stard', 'bast*rd', 'bastardly'],
'f*** you': ['fuck you', 'f u', 'f*** u', 'fuk you'],
'f*** off': ['fuck off', 'f off', 'f*** right off'],
'piece of shit': ['piece of s***', 'piece of sh*t', 'piece of sh!t'],
'this is bullshit': ['this is bullsh**', 'this is bs', 'this is bull sh*t'],
'f*** your rules': ['fuck your rules', 'f*** the rules', 'screw your rules'],
'f*** your advice': ['fuck your advice', 'screw your advice', 'your advice sucks'],
'f*** this chatbot': ['fuck this chatbot', 'this chatbot sucks', 'this bot is trash'],
'you are useless': ['you’re useless', 'you suck', 'you’re f***ing useless'],
'this sucks': ['this is stupid', 'this is dumb', 'this is trash'],
'you’re dumb': ['you’re stupid', 'you’re slow', 'you’re thick'],
'shut up': ['be quiet', 'stop talking', 'zip it'],
'i hate this': ['i hate this chatbot', 'this is awful', 'this is terrible'],
'you’re annoying': ['so annoying', 'you’re irritating', 'you’re frustrating'],
'this is a nightmare': ['this is a f***ing nightmare', 'this is hell', 'this is chaos'],
'this is ridiculous': ['this is f***ing ridiculous', 'this is absurd', 'this is nonsense'],
'you’re a joke': ['you’re a failure', 'you’re pathetic', 'you’re a clown'],
'you’re incompetent': ['you’re clueless', 'you’re incapable', 'you’re hopeless'],
'you’re a piece of s***': ['you’re trash', 'you’re garbage', 'you’re worthless'],
'this is a waste of time': ['this is pointless', 'this is useless', 'this is boring'],
'i’m done with this': ['i’m fed up', 'i’m outta here', 'i’m sick of this'],
'you’re not helpful': ['you’re no help', 'you’re useless', 'you’re not useful'],
'you’re not smart': ['you’re dumb', 'you’re slow', 'you’re clueless'],
'you’re not funny': ['you’re boring', 'you’re dry', 'you’re lame'],
'you’re not listening': ['you’re ignoring me', 'you don’t get it', 'you’re deaf'],
'you’re a waste of space': ['you’re pointless', 'you’re irrelevant', 'you’re in the way'],
'f*** this': ['fuck this', 'screw this', 'to hell with this'],
'f*** this s***': ['fuck this shit', 'screw this crap', 'to hell with this mess']

}

match_score = 0

for keyword in keywords:
    keyword = keyword.lower()

    if keyword in normalized_input:
        match_score += 1

    for synonym in keyword_synonyms.get(keyword, []):
        if synonym in normalized_input:
            match_score += 1

# Global variable to track current intent
current_intent = None
last_intent = None


# Conversation state tracking
conversation_state = {
    'intent': None,
    'last_user_input': None,
    'last_bot_response': None
}

# Defining functions

def correct_spelling(text):
    try:
        blob = TextBlob(text)
        corrected = blob.correct()
        return str(corrected)
    except Exception as e:
        print("Correction error:", e)
        return text  # fallback to original if correction fails

def calculate_match_score(user_input, intent_keywords, keyword_synonyms):
    match_score = 0
    user_input = user_input.lower()

    for keyword in intent_keywords:
        keyword = keyword.lower()

        if keyword in user_input:
            match_score += 1

        synonyms = keyword_synonyms.get(keyword, [])
        for synonym in synonyms:
            if synonym in user_input:
                match_score += 1
            else:
                distance = edit_distance(user_input, synonym)
                max_len = max(len(user_input), len(synonym))
                similarity = (distance / max_len) * 100
                if similarity < 70:
                    match_score += 0.5

    return match_score

def preprocess_input(user_input):
    return word_tokenize(user_input.lower())

def expand_input(tokens, keyword_synonyms):
    expanded = set(tokens)
    for token in tokens:
        if token in keyword_synonyms:
            expanded.update(keyword_synonyms[token])
    return expanded

def extract_keywords(user_input):
    # Tokenize user input
    tokens = word_tokenize(user_input)

def extract_pos_features(text):
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    features = {}

    for word, tag in tagged:
        if tag.startswith('NN') or tag.startswith('VB'):  # Nouns and Verbs
            features[word] = True

    return features

def count_content_words(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    content_words = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']]
    return len(content_words)


# Uses TF-based cosine similarity to match user input to the most semantically similar response from a list
def get_best_match(input_text, responses):
    vectorizer = CountVectorizer().fit_transform([input_text] + responses)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    best_index = similarities.argmax()
    return responses[best_index]

#  Stopwords Removal + Tokenization
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w not in stopwords.words('english')]
    return filtered

# Detecting the mood of the user's message and tailoring the reply accordingly.
    
def analyze_sentiment(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Range: [-1.0, 1.0]
    return {'compound': polarity}


keyword_synonyms = {}
for intent, keywords_list in keywords.items():
    keyword_synonyms[intent] = []
    for keyword in keywords_list:
        keyword = keyword.lower() 
        if keyword in keyword_synonyms:
            keyword_synonyms[intent].extend(keyword_synonyms[keyword])
        keyword_synonyms[intent].append(keyword.lower()) 
if any(greeting in user_input.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
    current_intent = 'greetings'

# Using the generated synonyms in intent identification logic
for intent, synonyms_list in keyword_synonyms.items():
    for synonym in synonyms_list:
            if synonym in user_input.lower():
                current_intent = intent
                break


# Defining features
def extract_features(text):
    if not isinstance(text, str):
        text = str(text)
    if pd.isna(text):
        text = ""
    
    tokens = word_tokenize(text.lower())
    unigram_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    bigram_tokens = [' '.join(bg) for bg in bigrams(tokens) if all(word.isalpha() and word not in stop_words for word in bg)]
    features = {token: True for token in unigram_tokens + bigram_tokens}
    return features


# Loading CSV files into DataFrames for each intent and preparing training data for model training and response generation

csv_files = {
    'admissions': 'CHAT BOT ADMISSIONS DATA.csv',
    'fees': 'CHAT BOT FEES DATA.csv',
    'physical_appointment': 'CHAT BOT PHYSICAL APPOINTMENT DATA.csv',
    'multi_purpose_hall': 'CHAT BOT MULTIPURPOSE HALL DATA.csv',
    'general_information': 'CHAT BOT GENERAL INFORMATION DATA.csv',
    'contact_information': 'CHAT BOT CONTACT INFORMATION DATA.csv',
    'careers': 'CHAT BOT CAREER DATA.csv',
    'business_partnership': 'CHAT BOT BUSINESS PARTNERSHIP DATA.csv',
    'feedback': 'CHAT BOT FEEDBACK DATA.csv',
    'greetings': 'CHAT BOT GREETINGS DATA.csv',
    'profanity': 'CHAT BOT PROFANITY DATA.csv'
}

dataframes = {}
training_data = []

for intent, file in csv_files.items():
    df = pd.read_csv(file)

    if 'response' not in df.columns:
        raise ValueError(f"Missing 'response' column in {file}")

    dataframes[intent] = df

    for _, row in df.iterrows():
        text = row['intent']  
        training_data.append((text, intent))  # Store raw text and intent

# Process text data using extract_features
processed_training_data = [(extract_features(text), intent) for text, intent in training_data]

# Train the classifier
classifier = NaiveBayesClassifier.train(processed_training_data)

# Reading responses from dataframe
filtered_responses = df[df['response'].apply(count_content_words) >= 3]['response'].tolist()
response_bank = df['response'].tolist()

def load_and_train(csv_files):
    dataframes = {}
    training_data = []

    for intent, file in csv_files.items():
        df = pd.read_csv(file)
        if 'response' not in df.columns:
            raise ValueError(f"Missing 'response' column in {file}")
        dataframes[intent] = df
        for _, row in df.iterrows():
            text = row['intent']
            training_data.append((extract_features(text), intent))

    classifier = NaiveBayesClassifier.train(training_data)
    return classifier, dataframes
classifier, dataframes = load_and_train(csv_files)

# TRAINING THE DATA FOR PREDICTED RESPONSES

training_data = []

for intent_label, df in dataframes.items():
    for _, row in df.iterrows():
        user_text = row['intent']  # This is the example input
        training_data.append((user_text, intent_label))  # (text, intent)


# Prepare training data for classifier
training_set = [(extract_features(text), intent) for text, intent in training_data if isinstance(text, str) and text.strip() != ""]

# Train the classifier
classifier = NaiveBayesClassifier.train(training_set)

# Matching user input to the most relevant intent using keyword, synonym, and fuzzy logic scoring

def match_intent(user_input):
    normalized_input = user_input.lower()
    best_match = None
    max_match_score = 0

    for intent, intent_keywords in keywords.items():
        match_score = 0

        for keyword in intent_keywords:
            keyword = keyword.lower()

            # Direct phrase match
            if keyword in normalized_input:
                match_score += 1

            # Synonym match
            synonyms = keyword_synonyms.get(keyword, [])
            for synonym in synonyms:
                synonym = synonym.lower()
                if synonym in normalized_input:
                    match_score += 1
                else:
                    # Optional fuzzy match for phrases
                    distance = edit_distance(normalized_input, synonym)
                    max_len = max(len(normalized_input), len(synonym))
                    similarity = (distance / max_len) * 100
                    if similarity < 20:
                        match_score += 0.8

        if match_score > max_match_score:
            max_match_score = match_score
            best_match = intent

    # Confidence score based on how many keywords matched
    confidence_score = max_match_score / len(keywords[best_match]) if best_match else 0
    return best_match, confidence_score

# Predicting the intent using nltk.classfy Naive Bayes
def predict_intent(user_input):
    global classifier
    if classifier is None:
        raise ValueError("Classifier is not trained.")
    features = extract_features(user_input)
    predicted = classifier.classify(features)
    prob_dist = classifier.prob_classify(features)
    confidence = prob_dist.prob(predicted)
    return predicted, confidence


#Matching user input with responses

def match_response(user_input, responses, max_threshold=5, min_threshold=1):
    user_words = set(user_input.lower().split())

    for threshold in range(max_threshold, min_threshold - 1, -1):
        best_match = None
        highest_score = 0

        for keywords, reply in responses.items():
            keyword_set = set(keywords.lower().split())
            score = len(user_words & keyword_set)

            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = reply

        if best_match:
            return best_match

    return "Sorry, I couldn't find a good match."

fallback_response = "I'm sorry, I didn't catch that. Could you please rephrase your question?"

# Resolving match intent, predict intent and match response

def resolve_user_input(user_input, responses, intent_threshold=0.5, response_threshold=0.5, overall_threshold=0.5):
    matched_intent, match_conf = match_intent(user_input)
    predicted_intent, predict_conf = predict_intent(user_input)
    direct_response = match_response(user_input, responses)
    
    # Estimate response confidence
    response_conf = 1 if direct_response and "Sorry" not in direct_response else 0
    
    # Use direct response if it's confident enough
    if response_conf >= response_threshold:
        return { 
            "response": direct_response, 
            "source": "match_response",
            "confidence": response_conf 
        }
    
    # Choose best intent based on confidence
    if match_conf >= intent_threshold and match_conf >= predict_conf:
        final_intent = matched_intent
        final_conf = match_conf
        source = "match_intent"
    else:
        final_intent = predicted_intent
        final_conf = predict_conf
        source = "predict_intent"
    
    # Check overall confidence threshold
    if final_conf < overall_threshold:
        return {
            "response": "Sorry, I didn't quite understand that. Could you please rephrase your question?",
            "confidence": final_conf,
            "source": "low_confidence"
        }
    
    # Get response from intent
    response = get_response(final_intent)
    return { 
        "response": response, 
        "intent": final_intent, 
        "confidence": final_conf, 
        "source": source 
    }

def generate_response(intent):
    global last_intent
    if intent == "unknown" and last_intent:
        intent = last_intent
    else:
        last_intent = intent

    responses = df.loc[df['intent'] == intent, 'response']
    if not responses.empty:
        return responses.sample(n=1).iloc[0]
    else:
        return "Sorry, I didn't quite catch that. Could you rephrase your question?"
            
# Processing the text (Using SpaCy)
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Extracting entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Defining the handlers for each intent

def handle_admissions_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    midterm_keywords = {'mid-term', 'midterm'}
    inquiry_keywords = {'why', 'feasible', 'possible', 'available'}

    stemmed_midterm = {stemmer.stem(word) for word in midterm_keywords}
    stemmed_inquiry = {stemmer.stem(word) for word in inquiry_keywords}

    if stemmed_midterm & stemmed_input and stemmed_inquiry & stemmed_input:
        return (
            "Mid-term admissions may not be feasible due to curriculum progression and class dynamics. "
            "However, we can discuss possible exceptions."
        )

    df = dataframes['admissions']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_fees_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    fee_keywords = {'fee', 'fees', 'bill', 'tuition', 'cost', 'payment'}
    stemmed_keywords = {stemmer.stem(word) for word in fee_keywords}

    if stemmed_keywords & stemmed_input:
        return (
            "Our fees cover tuition, books/stationery, uniform pack, utilities, exams and records, "
            "maintenance/development, and PTA levy. The cost will depend on the class your child is admitted into. "
            "The approximate fee for new intakes is between ₦120,000 to ₦200,000."
        )

    df = dataframes['fees']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_physical_appointment_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    appointment_keywords = {'require', 'need', 'examination', 'checkup', 'medical'}
    stemmed_keywords = {stemmer.stem(word) for word in appointment_keywords}

    if stemmed_keywords & stemmed_input:
        return (
            "For kids above 2 years old, a physical examination is required before admission."
        )

    df = dataframes['physical_appointment']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_multi_purpose_hall_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    hall_keywords = {'book', 'availability', 'reserve', 'schedule', 'event'}
    stemmed_keywords = {stemmer.stem(word) for word in hall_keywords}

    if stemmed_keywords & stemmed_input:
        return (
            "Our event hall is available for booking! Please contact our events coordinator "
            "to check availability and learn about the booking process."
        )

    df = dataframes['multi_purpose_hall']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_general_information_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    keywords = {'mission', 'values', 'vision', 'purpose', 'principles'}
    stemmed_keywords = {stemmer.stem(word) for word in keywords}

    if stemmed_keywords & stemmed_input:
        return (
            "Our mission is to provide quality education and foster growth in a supportive environment. "
            "We value curiosity, creativity, and community."
        )

    df = dataframes['general_information']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_contact_information_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    contact_keywords = {'phone', 'number', 'contact', 'call', 'reach'}
    stemmed_keywords = {stemmer.stem(word) for word in contact_keywords}
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    if stemmed_keywords & stemmed_input:
        return "You can reach us at 08139060571."

    df = dataframes['contact_information']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_careers_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    career_keywords = {'job', 'vacancy', 'employment', 'position', 'career', 'recruitment'}
    stemmed_keywords = {stemmer.stem(word) for word in career_keywords}
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    if stemmed_keywords & stemmed_input:
        return (
            "We're always looking for passionate educators to join our team! If you're interested in exploring job opportunities, you can submit your application by emailing your resume and a brief introduction, or preferably by visiting our school in person to submit a physical copy. We prioritize local candidates, as we believe it fosters a better work environment and community connection. We look forward to reviewing your application and discussing how you can contribute to our school community!"
        )

    df = dataframes['careers']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_business_partnership_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    partnership_keywords = {'partnership', 'collaboration', 'sponsorship', 'affiliate', 'cooperate'}
    stemmed_keywords = {stemmer.stem(word) for word in partnership_keywords}
    stemmed_input = {stemmer.stem(word) for word in filtered_tokens}

    if stemmed_keywords & stemmed_input:
        return (
            "We welcome business partnerships and sponsorships. Please contact us to discuss potential "
            "collaboration opportunities and how we can work together to support our school community."
        )

    df = dataframes['business_partnership']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_feedback_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    feedback_keywords = {'feedback', 'suggestion', 'comment', 'review', 'opinion'}
    if any(stemmer.stem(word) in feedback_keywords for word in filtered_tokens):
        return (
            "We value your feedback and suggestions. Please share your thoughts with us, "
            "and we will do our best to address any concerns or improve our services. "
            "Your input is important to us!"
        )

    df = dataframes['feedback']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

def handle_greetings_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    greetings_keywords = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'}
    if any(word in filtered_tokens for word in greetings_keywords):
        return 'Hello! How can I be of service?!'

    df = dataframes['greetings']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."
    
def handle_profanity_input(input_text):
    tokens = word_tokenize(input_text.lower())
    filtered_tokens = [t for t in tokens if t not in stop_words]

    profanity_keywords = {
        'fuck', 'shit', 'bastard', 'idiot', 'damn', 'hell', 'asshole', 'dick', 'crap', 'fool'
    }
    if any(word in filtered_tokens for word in profanity_keywords):
        return "Let's keep things respectful. I'm here to help if you need assistance."

    df = dataframes['profanity']
    for min_words in range(5, 0, -1):
        filtered = df[df['response'].apply(lambda x: len(str(x).split()) >= min_words)]
        if not filtered.empty:
            return get_best_match(input_text, filtered['response'].tolist())

    return "I'm sorry, I couldn't find a suitable response."

df = dataframes.get(intent)

if df is not None and 'response' in df.columns:
    response = df['response'].iloc[0]  
else:
    response = "I'm not sure how to respond to that."


def get_response(user_input):
    global current_intent

    # Exit handling
    if user_input.lower() in ['bye', 'thanks', 'thank you', 'goodbye']:
        current_intent = None
        return "Thanks for chatting! Have a great day. 👋"

    # Match intent
    intent, confidence = match_intent(user_input)

    # Use intent if matched, regardless of confidence
    if intent:
        current_intent = intent
    elif current_intent:
        intent = current_intent
    else:
        sentiment = sia.polarity_scores(user_input)
        if sentiment['compound'] > 0.5:
            return "You're enthusiastic about our school! How can I assist you today?"
        elif sentiment['compound'] < -0.5:
            return "Sorry to hear that you're not satisfied. Can you please provide more details so I can assist you better?"
        else:
            return "Sorry, I didn't quite catch that. Could you rephrase your question?"
        
    # Route to appropriate handler
    handlers = {
        'admissions': handle_admissions_input,
        'fees': handle_fees_input,
        'physical_appointment': handle_physical_appointment_input,
        'multi_purpose_hall': handle_multi_purpose_hall_input,
        'general_information': handle_general_information_input,
        'contact_information': handle_contact_information_input,
        'careers': handle_careers_input,
        'business_partnership': handle_business_partnership_input,
        'feedback': handle_feedback_input,
        'greetings': handle_greetings_input,
        'profanity': handle_profanity_input
    }

    return handlers.get(intent, lambda x: "I'm not sure how to respond to that.")(user_input)


#  Running the code

given_responses = {}
error_count = 0
max_attempts = 3
exit_commands = [
    # Direct exits
    'exit', 'quit', 'bye', 'goodbye', 'see you', 'see ya', 'later', 'ciao', 'bye for now',

    # Gratitude-based exits
    'thanks', 'thank you', 'appreciate it', 'cheers', 'much obliged',

    # Casual closers
    'talk later', 'chat later', 'catch you later', 'see you soon', 'peace out',
    'I’m done', 'that’s all', 'I’m good', 'I’m okay now',

    # Polite/formal exits
    'that will be all', 'I have to go', 'I’ll get back to you', 'we’ll talk soon',
    'I’ll reach out later', 'let’s continue this later', 'I’ll follow up',

    # Emoji or shorthand exits
    '👋', 'ttyl', 'brb', 'gtg', 'bye bye', 'ok bye'
]
while True:
    user_input = input("You: ")
    if any(edit_distance(user_input.lower(), command) / max(len(user_input), len(command)) * 100 < 30 for command in exit_commands):
        print("Evergreen Admin: Thanks for chatting! Have a great day. 👋")
        break
    response = get_response(user_input)
    print("Evergreen Admin:", response)
    if response == "Sorry, I didn't quite catch that. Could you rephrase your question?":
        error_count += 1
    if error_count >= max_attempts:
        print("Evergreen Admin: I apologize, but it seems we're having trouble understanding each other. Let's end our chat here. Feel free to start a new conversation if you need assistance in the future! 👋")
        break



# End of code


