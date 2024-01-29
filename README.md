# DevGPT-Study

## Abstract
The release of large language models (LLMs) like ChatGPT has revolutionized software development. Numerous studies are exploring the generated response quality of ChatGPT, the effectiveness of different prompting techniques, and its performance in programming contests, among other aspects. However, there is limited information regarding the practical usage of ChatGPT by software developers. This data mining challenge focuses on DevGPT, a curated dataset of developer-ChatGPT conversations encompassing prompts with ChatGPT’s responses, including code snippets. Our paper leverages this dataset to investigate  (RQ1) whether ChatGPT generates Python \& Java code with quality issues; (RQ2) whether ChatGPT-generated code is merged into a repository, and, if it does, to what extent developers change them; and (RQ3) what are the main use cases for ChatGPT besides code generation. We found that ChatGPT-generated code suffers from using undefined/unused variables and improper documentation. They are also suffering from improper resources and exception management-related security issues. Our results show that ChatGPT-generated codes are hardly merged, and they are significantly modified before merging.  Instead, based on an analysis of developers' discussions and the developer-ChatGPT chats, we found that developers use this model for every stage of software development and leverage it to learn about new frameworks and development kits.

## File/Folder Structure

- RQ1: Contains the code for RQ1
- RQ2_PR_Analysis.csv: Contains the data for RQ2
- RQ3_Results.csv: Contains the data for RQ3

## How to run the code

### Installation
We used Pylint and Bandit for our analysis. To install pylint/Bandit, run the following command:
```
pip install pylint
pip install bandit
```
In both cases, you may create a virtual environment and install the packages in it.

- Promptparser.py: This file contains the code to get Python and Java code from the *ChatgptSharing->Conversations->ListOfCode* and put in the right folder. 
- run_mvn.py: This file contains the code to run maven on the Java code and get the compilation errors.
- Java_maven: This folder contains the fixed Java code and the pom.xml file to run maven on the Java code.

We used the following command to run Pylint on the Python codes:
```
pylint Python  --output-format=json > pylint.json
```

We used the following command to run Bandit on the Python codes:
```
bandit -r ./Python -f json -o ./bandit.json
```

- CSV_Convertor.ipynb: This file contains the code to convert the JSON files to CSV files.