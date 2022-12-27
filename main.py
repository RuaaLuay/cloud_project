# This is a sample Python script.

import tokenize
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
from io import BytesIO
from PyPDF2 import PageObject
import os
import argparse
import re
from nltk.stem import PorterStemmer
from sklearn.tree import export_graphviz
import graphviz
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split
import fitz
import PyPDF2
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import time
import glob
from PyPDF2 import PdfFileReader
import os




def title(path):
    with open(path, "rb") as file:
        pdf=PdfFileReader(file)
        info= pdf.getDocumentInfo()
        page_numbers=pdf.getNumPages()
        if info.title==None:
            return "z"
        return info.title


def sort(dir_name):
    # List all the files in the current directory
    files = os.listdir(dir_name)
    # Sort the files by title (using metadata)
    sorted_files = sorted(files, key=lambda x: title(os.path.join(dir_name,x)))
    # Print the sorted list of files
    return sorted_files

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Call the function and pass the folder name as an argument

def search(s, folder_path):
    start_time = time.time()
    files = glob.glob('highlighted*')
    # Delete each file
    for f in files:
        os.remove(f)
    documents = []
    count = 0
    isHighlighed = 0
    for root, dirs, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        for file in files:
            file_path = os.path.join(root, file)
            # Add the file name to the list
            documents.append(file_path)
    # Iterate through the documents
    for document in documents:
        # Open the document
        doc = fitz.open(document)
        # Search for the text on each page
        for page in doc:
            # Search for the text in the page
            start = page.search_for(s, quads=True)
            if len(start) != 0:
                # Add a highlight annotation to the matched text
                page.add_highlight_annot(start)
                # Save the document with the highlight annotations
                last_ = document.split('\\')
                last_w = last_[-1]
                doc.save(f'highlighted_{last_w}')
                isHighlighed = 1
        if(isHighlighed == 1):
            count = count + 1
            isHighlighed =0
    print("Number of documents: " + str(len(documents)) + ". Number of documents that contain "+s+"= "+ str(count))
    end_time = time.time()
    TotalTime = end_time - start_time
    print("Total time:", TotalTime, "seconds")

def build_predefined_treeTest2(folder_path):
    # Read in the PDF documents and extract the text
    texts = [] #label, doc
    labels = []
    text =''
    for root, dirs, files in os.walk(folder_path):
        # Get the name of the current folder
        folder_name = os.path.basename(root)
        # Iterate over all files in the current folder
        for file in files:
            # Check if the file is a PDF file
            if file.endswith('.pdf'):
                # Open the PDF file
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    # Create a PDF object
                    pdf = PyPDF2.PdfFileReader(f)
                    text = ''
                    for page in range(pdf.getNumPages()):
                        text += pdf.getPage(page).extractText()
            texts.append(text)
            labels.append(folder_name)
    vectorizer = CountVectorizer(stop_words='english')
    # Generate the count vectors
    X = vectorizer.fit_transform(texts).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=0)
    print(X_train)
    print(y_train)
    #classifier = DecisionTreeClassifier(max_depth=10, criterion="entropy")
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)  # intialise data of lists.
    y_pred = classifier.predict(X_test)
    joblib.dump(classifier, 'classifier.pkl')
    accuracy = (y_pred == y_test).mean()
    print(accuracy)
    print(y_pred)

def predict(file_path2):
    start_time = time.time()
    texts = []
    with open(file_path2, 'rb') as f:
        pdf = PyPDF2.PdfFileReader(f)
        text = ''
        for page in range(pdf.getNumPages()):
            text += pdf.getPage(page).extractText()
            texts.append(pdf.getPage(page).extractText())
    # Load the data you want to make predictions on
    model = joblib.load('classifier.pkl')
    vectorizer = CountVectorizer(stop_words='english', max_features=1400)
    X = vectorizer.fit_transform([text]).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    X = vectorizer.fit_transform([text]).toarray()
    print(X)
    pre = model.predict(X)
    print(pre)
    end_time = time.time()
    TotalTime = end_time - start_time
    print("Total time:", TotalTime, "seconds")
    doc = fitz.open(file_path2)
    last_ = file_path2.split('\\')
    last_w = last_[-1]
    doc.save(f'C:\\Users\\JIT\\Downloads\\Cloud\\{pre[0]}\\{last_w}')
    #X_new = selector.transform(X)
    # Load the model from the file
    # Make predictions on the data
    #predictions = model.predict(X_new)
    # Generate the count vectors


    #features = vectorizer.fit_transform([text])

    # Extract the features from a new PDF document
    #new_pdf_features = features

    # Classify the new PDF document
    #prediction = clf.predict(new_pdf_features)
    #print_hi(prediction)

   # print_hi(predictions)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   #print_hi('hello')
   #build_predefined_treeTest2('C:\\Users\\JIT\\Downloads\\Cloud\\')
   #predict('C:\\Users\\JIT\\Downloads\\Cloud\\Politics\\Perceptions_of_Women_Living_with_AIDS_in.pdf')
   #search("ooooooo", 'C:\\Users\\JIT\\Downloads\\Cloud\\')
   print(sort('C:\\Users\\JIT\\Downloads\\Cloud\\Politics'))


   #open_files_in_folder('C:\\Users\\JIT\\Downloads\\Cloud\\Politics','and')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
