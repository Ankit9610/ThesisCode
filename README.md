# ThesisCode
Technical Part of My Master Thesis.
Steps to follow to run this project from scratch in any laptop. 

Required:-
(i)Good Internet connection.
(ii) Python Installed. 


My Set up :- 
(i)VS Code 
(ii)Windows System.

Step 1 :- Open the project in VS Code.Then Click on Terminal and then New Terminal

Step 2 :- Run below command on Terminal. 

(i)Create a Virtual Environment

python -m venv venv

(ii)Activate it.

.\venv\Scripts\activate

(iii)Upgrade pip

python -m pip install --upgrade pip

(iv)Install Required Python Libraries

pip install numpy pandas matplotlib seaborn scikit-learn nltk spacy textstat scipy


(v)Download the Required spaCy Model

python -m spacy download en_core_web_sm

(vi)Download Required NLTK Resources.

python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"


Step 3 :- Put all the code in a file.  Then Run that python file.


Uses :- 
This programm is the analysis of human written text and Ai generated text of RAID dataset.This python program perform below things. 

Load Dataset 
Data quality checks
Document length analysis
Text cleaning
Readability analysis
Sentiment analysis
TF-IDF analysis
Bigram analysis
Bigram language-model perplexity
Burstiness analysis
Truncated SVD dimensionality reduction
t-SNE dimensionality reduction
Statistical comparison between human and AI documents
 
 Output:- 
 (i)Terminal Output :- 

Dataset information
Missing values
Class balance
Domain balance
Document length statistics
Readability results
Sentiment results
Top TF-IDF terms
Top bigrams
Perplexity results
Burstiness results
Dimensionality-reduction results
Statistical comparison results





 

 (ii)The code creates a folder and saves all graph in that folder.
