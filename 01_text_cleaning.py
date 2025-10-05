# Remove extra spaces between words
text = "Hello,      World!     I came from 2030"

cleaned_text1 = " ".join(text.split())
print(cleaned_text1)

# Convert all characters to lowercase
text = "Hello,      World!     I came from 2030"

cleaned_text2 = text.lower()
print(cleaned_text2)

# Remove punctuation marks from the text
import string

text = "Hello,      World!     I came from 2030"

cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(cleaned_text3)

# Remove all non-alphanumeric characters using regex
import re

text = "Hello,      World!     I came from 2030"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","", text)
print(cleaned_text4)

# Correct spelling mistakes in the text
# (Requires installation of TextBlob)
pip install textblob

from textblob import TextBlob

text = "Hello,      Wirld!     I ceme from 2030"

cleaned_text5 = str(TextBlob(text).correct())
print(cleaned_text5)

# Remove HTML tags and extract clean text content
from bs4 import BeautifulSoup

html_text = "<div>Hello,      World!     I came from 2030</div>"

cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(cleaned_text6)