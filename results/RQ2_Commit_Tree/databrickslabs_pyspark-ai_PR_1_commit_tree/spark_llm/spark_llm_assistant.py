import re
from typing import Callable, Optional, List
from urllib.parse import urlparse

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain import LLMChain, GoogleSearchAPIWrapper, BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from pyspark.sql import SparkSession, DataFrame
from tiktoken import Encoding

from spark_llm.cache import Cache
from spark_llm.llm_chain_with_cache import LLMChainWithCache
from spark_llm.prompt import (
    SEARCH_PROMPT,
    SQL_PROMPT,
    EXPLAIN_DF_PROMPT,
    TRANSFORM_PROMPT,
    PLOT_PROMPT,
    VERIFY_PROMPT
)
from spark_llm.search_tool_with_cache import SearchToolWithCache


class SparkLLMAssistant:
    _HTTP_HEADER = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        llm: BaseLanguageModel,
        web_search_tool: Optional[Callable[[str], str]] = None,
        spark_session: Optional[SparkSession] = None,
        enable_cache: bool = True,
        encoding: Optional[Encoding] = None,
        max_tokens_of_web_content: int = 3000,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SparkLLMAssistant object with the provided parameters.

        :param llm: LLM instance for selecting web search result
                                 and writing the ingestion SQL query.
        :param web_search_tool: optional function to perform web search,
                                Google search will be used if not provided
        :param spark_session: optional SparkSession, a new one will be created if not provided
        :param encoding: optional Encoding, cl100k_base will be used if not provided
        :param max_tokens_of_web_content: maximum tokens of web content after encoding
        """
        self._spark = spark_session or SparkSession.builder.getOrCreate()
        self._llm = llm
        self._web_search_tool = web_search_tool or self._default_web_search_tool
        if enable_cache:
            self._cache = Cache()
            self._web_search_tool = SearchToolWithCache(
                self._web_search_tool, self._cache
            ).search
        else:
            self._cache = None
        self._encoding = encoding or tiktoken.get_encoding("cl100k_base")
        self._max_tokens_of_web_content = max_tokens_of_web_content
        self._search_llm_chain = self._create_llm_chain(prompt=SEARCH_PROMPT)
        self._sql_llm_chain = self._create_llm_chain(prompt=SQL_PROMPT)
        self._explain_chain = self._create_llm_chain(prompt=EXPLAIN_DF_PROMPT)
        self._transform_chain = self._create_llm_chain(prompt=TRANSFORM_PROMPT)
        self._plot_chain = self._create_llm_chain(prompt=PLOT_PROMPT)
        self._verbose = verbose

    def _create_llm_chain(self, prompt: BasePromptTemplate):
        if self._cache is None:
            return LLMChain(llm=self._llm, prompt=prompt)

        return LLMChainWithCache(llm=self._llm, prompt=prompt, cache=self._cache)

    @staticmethod
    def _extract_view_name(query: str) -> str:
        """
        Extract the view name from the provided SQL query.

        :param query: SQL query as a string
        :return: view name as a string
        """
        pattern = r"^CREATE(?: OR REPLACE)? TEMP VIEW (\S+)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"The provided query: '{query}' is not valid for creating a temporary view. "
                "Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'"
            )
        return match.group(1)

    @staticmethod
    def _generate_search_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The best search results should contain as many as possible of these info: {','.join(columns)}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _generate_sql_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The result view MUST contain following columns: {columns}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _default_web_search_tool(desc: str) -> str:
        search_wrapper = GoogleSearchAPIWrapper()
        return str(search_wrapper.results(query=desc, num_results=10))

    @staticmethod
    def _is_http_or_https_url(s: str):
        result = urlparse(s)  # Parse the URL
        # Check if the scheme is 'http' or 'https'
        return result.scheme in ["http", "https"]

    @staticmethod
    def _extract_code_blocks(text) -> List[str]:
        code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
        code_blocks = re.findall(code_block_pattern, text)
        if code_blocks:
            # If there are code blocks, strip them and remove language specifiers.
            extracted_blocks = []
            for block in code_blocks:
                block = block.strip()
                if block.startswith("python"):
                    block = block.replace("python\n", "", 1)
                elif block.startswith("sql"):
                    block = block.replace("sql\n", "", 1)
                extracted_blocks.append(block)
            return extracted_blocks
        else:
            # If there are no code blocks, treat the whole text as a single block of code.
            return [text]

    def log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _trim_text_from_end(self, text: str, max_tokens: int) -> str:
        """
        Trim text from the end based on the maximum number of tokens allowed.

        :param text: text to trim
        :param max_tokens: maximum tokens allowed
        :return: trimmed text
        """
        tokens = list(self._encoding.encode(text))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self._encoding.decode(tokens)

    def _get_url_from_search_tool(self, desc: str, columns: Optional[List[str]]) -> str:
        search_result = self._web_search_tool(desc)
        search_columns_hint = self._generate_search_prompt(columns)
        # Run the LLM chain to pick the best search result
        return self._search_llm_chain.run(
            query=desc, search_results=search_result, columns={search_columns_hint}
        )

    def _create_dataframe_with_llm(
        self, text: str, desc: str, columns: Optional[List[str]]
    ) -> DataFrame:
        clean_text = " ".join(text.split())
        web_content = self._trim_text_from_end(
            clean_text, self._max_tokens_of_web_content
        )

        sql_columns_hint = self._generate_sql_prompt(columns)

        # Run the LLM chain to get an ingestion SQL query
        llm_result = self._sql_llm_chain.run(
            query=desc, web_content=web_content, columns=sql_columns_hint
        )
        sql_query = self._extract_code_blocks(llm_result)[0]
        self.log(f"SQL query for the ingestion:\n {sql_query}\n")

        view_name = self._extract_view_name(sql_query)
        self.log(f"Storing data into temp view: {view_name}\n")
        self._spark.sql(sql_query)
        return self._spark.table(view_name)

    def _get_df_schema(self, df: DataFrame) -> str:
        return "\n".join([f"{name}: {dtype}" for name, dtype in df.dtypes])

    @staticmethod
    def _trim_hash_id(analyzed_plan):
        # Pattern to find strings like #59 or #2021
        pattern = r"#\d+"

        # Remove matching patterns
        trimmed_plan = re.sub(pattern, "", analyzed_plan)

        return trimmed_plan

    def _get_df_explain(self, df: DataFrame) -> str:
        raw_analyzed_str = df._jdf.queryExecution().analyzed().toString()
        return self._explain_chain.run(input=self._trim_hash_id(raw_analyzed_str))

    def create_df(self, desc: str, columns: Optional[List[str]] = None) -> DataFrame:
        """
        Create a Spark DataFrame by querying an LLM from web search result.

        :param desc: the description of the result DataFrame, which will be used for
                     web searching
        :param columns: the expected column names in the result DataFrame
        :return: a Spark DataFrame
        """
        url = desc.strip()  # Remove leading and trailing whitespace
        is_url = self._is_http_or_https_url(url)
        # If the input is not a valid URL, use search tool to get the dataset.
        if not is_url:
            url = self._get_url_from_search_tool(desc, columns)

        self.log(f"Parsing URL: {url}\n")
        try:
            response = requests.get(url, headers=self._HTTP_HEADER)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            self.log(f"HTTP error occurred: {http_err}")
            return
        except Exception as err:
            self.log(f"Other error occurred: {err}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        # If the input is a URL link, use the title of web page as the dataset's description.
        if is_url:
            desc = soup.title.string
        return self._create_dataframe_with_llm(soup.get_text(), desc, columns)

    def transform_df(self, df: DataFrame, desc: str) -> DataFrame:
        """
        This method applies a transformation to a provided Spark DataFrame, the specifics of which are determined by the 'desc' parameter.

        :param df: The Spark DataFrame that is to be transformed.
        :param desc: A natural language string that outlines the specific transformation to be applied on the DataFrame.

        :return: Returns a new Spark DataFrame that is the result of applying the specified transformation on the input DataFrame.
        """
        temp_view_name = "temp_view_for_transform"
        df.createOrReplaceTempView(temp_view_name)
        schema_str = self._get_df_schema(df)
        llm_result = self._transform_chain.run(
            view_name=temp_view_name, columns=schema_str, desc=desc
        )
        sql_query = self._extract_code_blocks(llm_result)[0]
        self.log(f"SQL query for the transform:\n{sql_query}")
        return self._spark.sql(sql_query)

    def explain_df(self, df: DataFrame) -> str:
        """
        This method generates a natural language explanation of the SQL plan of the input Spark DataFrame.

        :param df: The Spark DataFrame to be explained.

        :return: A string explanation of the DataFrame's SQL plan, detailing what the DataFrame is intended to retrieve.
        """
        explain_result = self._get_df_explain(df)
        # If there is code block in the explain result, ignore it.
        if "```" in explain_result:
            summary = explain_result.split("```")[-1]
            return summary.strip()
        else:
            return explain_result

    def plot_df(self, df: DataFrame) -> None:
        response = self._plot_chain.run(
            columns=self._get_df_schema(df), explain=self._get_df_explain(df)
        )
        self.log(response)
        codeblocks = self._extract_code_blocks(response)
        for code in codeblocks:
            exec(code)

    def verify_df(self, df: DataFrame, desc: str) -> None:
        """
        This method creates and runs test cases for the provided PySpark dataframe transformation function.

        :param df: The Spark DataFrame to be verified
        :param desc: A description of the expectation to be verified
        """
        llm_output = self._verify_chain.run(
            df=df,
            desc=desc
        )

        self.log(f"Generated code:\n{llm_output}")

        locals_ = {}
        exec(llm_output, {"df": df}, locals_)

        self.log(f"\nResult: {locals_['result']}")

    def activate(self):
        """
        Activates LLM utility functions for Spark DataFrame.
        """
        DataFrame.llm_transform = lambda df_instance, desc: self.transform_df(
            df_instance, desc
        )
        DataFrame.llm_explain = lambda df_instance: self.explain_df(df_instance)
        DataFrame.llm_plot = lambda df_instance: self.plot_df(df_instance)
        DataFrame.llm_verify = lambda df_instance, desc: self.verify_df(df_instance, desc)

    def commit(self):
        """
        Commit the staging in-memory cache into persistent cache, if cache is enabled.
        """
        if self._cache is not None:
            self._cache.commit()
