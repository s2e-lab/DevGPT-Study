# DevGPT-Study

This repository contains the code and data for the paper "Quality Assessment of ChatGPT Generated Code and their Use by Developers" accepted at the Mining Challenge Track of the 21st International Conference on Mining Software Repositories (MSR 2024).

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

For installing CodeQL, please follow the instructions from [here](https://docs.github.com/en/code-security/codeql-cli).

### Running the code
To get the result for Python, check this:

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

To get the result for Java, check this:

- CodeQL_Runner.ipynb: This file contains the code to run CodeQL on the Java code. It uses the script.sh file to run CodeQL on the Java code.
- Data_filteration.ipynb: This file contains the code to filter the results of CodeQL and get the final results.

## Citation
If you use our dataset or code, please cite our paper:
```
@inproceedings{siddiq2024devgpt,
  author={Siddiq, Mohammed Latif and Roney, Lindsay and Zhang, Jiahao and Santos, Joanna C. S.},
  booktitle={Proceedings of the 21st International Conference on Mining Software Repositories, Mining Challenge Track (MSR 2024)}, 
  title={Quality Assessment of ChatGPT Generated Code and their Use by Developers}, 
  year={2024}
}
```