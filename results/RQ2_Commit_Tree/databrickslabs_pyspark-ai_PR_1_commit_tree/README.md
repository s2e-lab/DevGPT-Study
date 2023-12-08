# LLM Assistant for Apache Spark

## Installation

```bash
pip install spark-llm
```

## Usage
### Initialization
```python
from langchain.chat_models import ChatOpenAI
from spark_llm import SparkLLMAssistant

llm = ChatOpenAI(model_name='gpt-4') # using gpt-4 can achieve better results
assistant=SparkLLMAssistant(llm=llm)
assistant.activate() # active partial functions for Spark DataFrame
```

### Data Ingestion
```python
auto_df=assistant.create_df("2022 USA national auto sales by brand")
auto_df.show(n=5)
```
| rank | brand     | us_sales_2022 | sales_change_vs_2021 |
|------|-----------|---------------|----------------------|
| 1    | Toyota    | 1849751       | -9                   |
| 2    | Ford      | 1767439       | -2                   |
| 3    | Chevrolet | 1502389       | 6                    |
| 4    | Honda     | 881201        | -33                  |
| 5    | Hyundai   | 724265        | -2                   |

### Plot
```python
auto_df.llm_plot()
```
![2022 USA national auto sales by brand](docs/_static/auto_sales.png)
### DataFrame Transformation
```python
auto_top_growth_df=auto_df.llm_transform("top brand with the highest growth")
auto_top_growth_df.show()
```
| brand    | us_sales_2022 | sales_change_vs_2021 |
|----------|---------------|----------------------|
| Cadillac | 134726        | 14                   |

### DataFrame Explanation
```python
auto_top_growth_df.llm_explain()
```

> In summary, this dataframe is retrieving the brand with the highest sales change in 2022 compared to 2021. It presents the results sorted by sales change in descending order and only returns the top result.

Refer to [example.ipynb](https://github.com/gengliangwang/spark-llm/blob/main/examples/example.ipynb) for more detailed usage examples.

### DataFrame Attribute Verification
```python
auto_top_growth_df.llm_verify("expect sales change percentage to be between -100 to 100")
```

```python
# Generated code:
def is_sales_change_valid(df) -> bool:
    # Check if the sales_change column exists in the DataFrame
    if 'sales_change_vs_2021' not in df.columns:
        return False

    # Filter rows where sales_change is between -100 and 100
    valid_rows = df.filter((df.sales_change_vs_2021 >= -100) & (df.sales_change_vs_2021 <= 100))

    # Check if all rows are valid
    if valid_rows.count() == df.count():
        return True
    else:
        return False

result = is_sales_change_valid(df)
```
> result: True

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Licensed under the Apache License 2.0.
