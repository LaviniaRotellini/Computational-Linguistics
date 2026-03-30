## Computational Linguistics and NLP project
Solving an authorship attribution task on a corpus composed of works by three authors: Victor Hugo, Charles Dickens and George Eliot. 
The analysis is divided in five tasks: 
- **Pre processing** (preprocessing.ipynb): counting and cleaning of paragraphs, creation of the dataset
- Classification using linear Support Vector Machine based on **linguistic information**, lexical information excluded (Task_1.ipynb)
- Classification using linear Support Vector Machine and **n-grams of words, tokens, characters** (Task_2.ipynb, Task_2_secondaParte.ipynb)
- Classification using linear Support Vector Machine and **word embeddings** (Task_3.ipynb)
- Fine tuning and classification using a **Transformer** model (Task_4.ipynb)

----
Other files: 
- utils_svm.py: code written in support of the main tasks applying Support Vector Machine
- utils.py: functions used to clean and prep the data
- report_Rotellini.pdf: the report written for the exam
