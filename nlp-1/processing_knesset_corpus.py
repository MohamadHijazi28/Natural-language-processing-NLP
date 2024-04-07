import os
import re
import string
import csv
import sys

import pandas as pd
from docx import Document


# get docx files from the folder_path
def get_docx_files(folder_path):
    docx_files = [folder for folder in os.listdir(folder_path)]
    return docx_files


# read file names
def read_file_names(folder_path):
    return get_docx_files(folder_path)


# split the file name to parts by '_'
def file_name_split(file_name):
    parts = file_name.split('_')
    return parts


# save and print the file name, knesset number and protocol type
def save_info_for_file(docx_files):
    files_list = []
    for docx_file in docx_files:
        parts = file_name_split(docx_file)
        if parts[1] == 'ptm':
            file_data = {'FileName ': docx_file, 'knessetNumber ': parts[0], 'CommitteeOrPlenary ': 'plenary'}
        else:
            file_data = {'FileName ': docx_file, 'knessetNumber ': parts[0], 'CommitteeOrPlenary ': 'committee'}

        files_list.append(file_data)

    for file_info in files_list:
        print(file_info)


# return knesset number and protocol type
def get_info_for_file(docx_file_name):
    parts = file_name_split(docx_file_name)
    if parts[1] == 'ptm':
        return parts[0], 'plenary'
    else:
        return parts[0], 'committee'


# the function search about the first talker and arrange the words on the dialog
def get_persons_words(docx_path):
    try:
        doc = Document(docx_path)
        dialog_list = []
        current_person = None
        last_person = None
        current_text = ''
        important = False
        empty_lines = 0
        first_person_flag = False
        specific_file_one = False
        specific_file_second = False
        specific_file_names = False

        for par in doc.paragraphs:
            par_words = par.text.split()
            if not first_person_flag:  # on all the files the first speaker is the 'היו"ר'
                if len(par_words) > 6:
                    if par_words[-4].endswith(':') and par_words[1] == 'יור':
                        first_person_flag = True
                    if par_words[0] == '<<' and par_words[-1] == '>>':
                        specific_file_one = True
                elif len(par_words) > 2 and par.text.endswith(':') and (par_words[0] == 'היו”ר' or par_words[0] == 'מ"מ'):
                    first_person_flag = True
                elif par.text.endswith(':') and par_words[0] == 'מר':
                    first_person_flag = True
                    specific_file_names = True
                elif par.text.endswith(':>') and par_words[0] == '<היו"ר':
                    first_person_flag = True
                    specific_file_second = True
                elif len(par_words) == 3 and par_words[0] == 'היו"ר':
                    first_person_flag = True
            if (first_person_flag and (
                    (par.text.endswith(':') and len(par_words) < 10) or (len(par_words) > 6 and par_words[-4].endswith(':'))
                    or par.text.endswith(':>')
                    or (len(par_words) == 3 and (par_words[0] == 'היו"ר' or par_words[0] == 'מר')))
                    or (len(par_words) >= 3 and par_words[0] == 'היו"ר')
                    and (',' not in par.text)):
                important = True
                par_contain_name = par.text
                if specific_file_one:
                    par_contain_name = ' '.join(par_words[3:-3])
                if specific_file_second:
                    par_contain_name = par.text[1:-1]
                if specific_file_names and par_words[0] == 'מר':
                    par_contain_name = ' '.join(par_words[1:])
                current_person = re.sub(r'\([^)]*\)', '', par_contain_name)
                empty_lines = 0
                person_name_len = len(current_person.split())
                if current_person != par.text:
                    person_name_len -= 1
                words = current_person.split()
                if person_name_len >= 4:
                    current_person = ' '.join(words[-2:])

                person_name = current_person
                if last_person is not None:
                    dialog_list.append({'person': last_person, 'text': current_text.strip()})
                current_text = ''
                last_person = current_person

            elif par.text == '':
                empty_lines += 1
                if empty_lines < 2:
                    current_text += par.text + ' '

            elif empty_lines >= 2:
                important = False
                empty_lines = 0

            elif empty_lines == 1 and sum(len(run.text.split()) for run in par.runs) == 1 and par.text.endswith(':'):
                important = False
            else:
                if important:
                    empty_lines = 0
                    current_text += par.text + ' '

        # saving the last person text
        dialog_list.append({'person': last_person, 'text': current_text.strip()})

        return dialog_list
        
    except Exception as e:
        print(f"Error reading document {docx_path}: {e}")
        return []

# collect all the text for each person on the dialog
def get_all_text_to_person(dialog_list):
    persons_text = {}
    for entry in dialog_list:
        person_name = entry['person']
        person_text = entry['text']
        if person_name not in persons_text:
            persons_text[person_name] = person_text
        else:
            persons_text[person_name] += '\n' + person_text

    return persons_text


# split the text to sentences, by some of the punctuations
def split_to_sentences(text):
    splitting_sentence_pattern = re.compile(r'[.!?]')
    dialog_sentences = []
    current_sentence = ''

    for word in text:
        current_sentence += word
        if splitting_sentence_pattern.match(word) or current_sentence.endswith("– – –"):
            dialog_sentences.append(current_sentence)
            current_sentence = ''

    # Remove leading/trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in dialog_sentences]

    return sentences


# after split the text to sentences, we want to clean and remove the sentences that: contain an english letter,
# or contain '– – –', or contain anything that is not on the text language(hebrew) - (not including digits)
def clean_sentence(sentence):
    try:
        hebrew_pattern = re.compile(r'[^\u0590-\u05FF0-9]+')
        dashes_pattern = re.compile(r'– – –')
        new_lines = re.compile(r'\n')
        if any(char.isalpha() and ord(char) < 128 for char in sentence):
            return False
        elif not bool(hebrew_pattern.search(sentence)):
            return False
        elif bool(dashes_pattern.search(sentence)):
            return False
        elif bool(new_lines.search(sentence)):
            return False

        return True
        
    except Exception as e:
        print(f"Error cleaning sentence: {e}")
        return False    

# Split a sentence to tokens
def word_tokenize(sentence):
    try:
        sentence_words = []
        current_word = ''

        for i in range(len(sentence)):
            char = sentence[i]
            if i < len(sentence) - 1:
                next_char = sentence[i + 1]
            else:
                next_char = ''

            if char.isalnum() or char == '_':
                current_word += char

            # Handle " symbol within words like - התשנ"ג
            elif char == '"' and current_word and current_word[-1].isalpha() and next_char.isalpha():
                current_word += char

            elif char in string.punctuation:
                if current_word:
                    # Checks if it's a comma between digits
                    if char == ',' and next_char == ' ':
                        sentence_words.append(current_word)
                        sentence_words.append(char)
                        current_word = ''
                    # Checks if it's a comma not between digits
                    elif char == ',' and next_char.isdigit() and current_word.replace(',', '').isdigit():
                        current_word += char
                    elif char == '%' and current_word[:-1].isdigit():
                        current_word += char
                    elif char == ':' and next_char.isdigit():
                        current_word += char
                    elif char == '\'' and current_word[-1].isalpha():
                        current_word += char
                    elif char == '/' and (current_word[-1].isalpha() or current_word[-1].isdigit()) and next_char.isdigit():
                        current_word += char
                    else:
                        sentence_words.append(current_word)
                        current_word = ''
                if ((char != ',' and char != '%' and char != '\'' and (char != ':' and next_char.isdigit())
                    and (char != '/' and next_char.isdigit())) or char == '.'):
                    sentence_words.append(char)

            elif current_word:
                sentence_words.append(current_word)
                current_word = ''

        if current_word:
            sentence_words.append(current_word)

        return sentence_words
        
    except Exception as e:
        print(f"Error tokenizing sentence: {e}")
        return []


# folder_path = input("Enter folder path: ")
folder_path = sys.argv[1]

files_names = os.listdir(folder_path)

# Specify the output CSV file path
# output_path = input("Enter output path: ")
output_path = sys.argv[2]
output_csv_path = output_path + "\\knesset_corpus.csv"

# Create a list to store data
data_list = []

# Iterate through the files
for index, file_name in enumerate(files_names):
    try:
        if file_name.endswith(".docx"):
            # get the knesset number and the protocol type of each file
            knesset_number, protocol_type = get_info_for_file(file_name)

            # result contain the dialog between the talker(just the dialog)
            result = get_all_text_to_person(get_persons_words(folder_path + '\\' + file_name))

            for person, collected_text in result.items():
                sentences = split_to_sentences(collected_text)

                for sentence in sentences:
                    if clean_sentence(sentence):
                        words = word_tokenize(sentence)
                        if len(words) >= 4:  # getting the sentences that tokens number on it is more than 4
                            # Append data to the list
                            data_list.append({
                                'protocol_name': file_name,
                                'knesset_number': knesset_number,
                                'protocol_type': protocol_type,
                                'speaker_name': person[:-1],
                                'sentence_text': ' '.join(words)
                            })
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        
    # Print progress
    progress = (index + 1) / len(files_names) * 100
    print(f"Progress: {progress:.0f}%")

# Write the data to CSV using pandas
df = pd.DataFrame(data_list)
df.to_csv(output_csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, quotechar="$")

print("Data has been successfully written to CSV:", output_csv_path)
