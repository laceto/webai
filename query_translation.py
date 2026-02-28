import logging

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output models
# Each model is registered as a LangChain tool; its Field description is sent
# to the LLM as the tool parameter description.
# ---------------------------------------------------------------------------

class DecomposedQuery(BaseModel):
    decomposed_query: str = Field(
        ...,
        description="A unique sub-query of the original input query.",
    )


class GeneralQuery(BaseModel):
    general_query: str = Field(
        ...,
        description="A unique broader and more abstract query of the original question.",
    )


class ParaphrasedQuery(BaseModel):
    # The class docstring is used by LangChain as the tool description sent to the LLM.
    """You have performed query expansion to generate a paraphrasing of a question."""
    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )


# ---------------------------------------------------------------------------
# Default few-shot examples — module-level constants so callers can inspect or
# override them per call via the few_shot_examples parameter on each function.
# ---------------------------------------------------------------------------

_DEFAULT_DECOMPOSE_EXAMPLES: list[dict] = [
    {
        "original_query": (
            "In what ways do the terms of this reinsurance agreement address "
            "the allocation of losses between the primary insurer and the reinsurer?"
        ),
        "new_queries": [
            "What specific clauses detail the loss allocation between the primary insurer and reinsurer?",
            "How are losses shared between the two parties according to this agreement?",
            "What criteria are used to determine how losses are allocated?",
            "Are there any caps on the amount of loss each party must cover?",
            "How does this agreement ensure transparency in loss allocation?",
        ],
    },
]

_DEFAULT_STEP_BACK_EXAMPLES: list[dict] = [
    {
        "original_query": "Impact of AI on job market",
        "new_queries": [
            "Technological evolution and its fundamental impact on workforce dynamics",
            "Socioeconomic transformations driven by artificial intelligence and automation",
            "Ethical considerations in the intersection of technological innovation and human labor",
        ],
    },
]

_DEFAULT_EXPAND_EXAMPLES: list[dict] = [
    {
        "original_query": "How is the premium calculated in this reinsurance agreement",
        "new_queries": [
            "What factors influence the calculation of the premium in this reinsurance agreement?",
            "Can you explain the methodology for calculating the premium in this agreement?",
            "What are the criteria used to determine the premium in the reinsurance agreement?",
        ],
    },
    {
        "original_query": "Who are the parties involved in the reinsurance agreement?",
        "new_queries": [
            "Can you list the parties that are part of the reinsurance agreement?",
            "Who are the entities involved in this reinsurance agreement?",
            "What parties are specified in the reinsurance agreement?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_query_chain(
    model: BaseChatModel,
    tool_type: type[BaseModel],
    template: str,
):
    """
    Build a LangChain LCEL batch chain for a query-translation tool type.

    Composes: PromptTemplate → LLM-with-tools → PydanticToolsParser.
    Input variables are auto-detected from the template string by PromptTemplate,
    so templates must use {variable_name} syntax only for PromptTemplate variables.
    Values that could contain curly braces (e.g., few_shot_str) must be passed
    as template variables, never via Python f-string interpolation.

    Parameters:
        model (BaseChatModel): LangChain chat model.
        tool_type (type[BaseModel]): Pydantic model used as the structured output tool.
        template (str): PromptTemplate string. Must contain {question}; may also
            contain {few_shot_str}. Integer-valued parameters (e.g., num_queries)
            are safe to bake in via f-string before calling this function.

    Returns:
        Runnable: A LangChain chain ready for .batch() calls.
    """
    parser = PydanticToolsParser(tools=[tool_type])
    llm_with_tools = model.bind_tools([tool_type])
    # from_template auto-detects input_variables from {placeholder} syntax.
    prompt = PromptTemplate.from_template(template)
    return prompt | llm_with_tools | parser


def create_input_objects(input_strings: list[str], **common_fields) -> list[dict]:
    """
    Convert a list of query strings into PromptTemplate-compatible input dicts.

    Each returned dict contains {"question": str} plus any extra fields passed
    as keyword arguments. Extra fields are constant across all batch items and
    are used to supply additional PromptTemplate variables (e.g., few_shot_str).

    Args:
        input_strings (list[str]): A non-empty list of query strings.
        **common_fields: Additional key-value pairs merged into every input dict.

    Returns:
        list[dict]: One {"question": str, ...} dict per input string.

    Raises:
        ValueError: If input_strings is not a list or is empty.
        TypeError: If any element of input_strings is not a string.
    """
    if not isinstance(input_strings, list):
        raise ValueError(
            f"input_strings must be a list, got {type(input_strings).__name__}."
        )
    if not input_strings:
        raise ValueError("input_strings must be a non-empty list.")
    if not all(isinstance(s, str) for s in input_strings):
        raise TypeError("All elements of input_strings must be strings.")
    return [{"question": s, **common_fields} for s in input_strings]


def format_few_shot_examples(
    few_shot_examples: list[dict],
    label: str = "Expanded Queries",
) -> str:
    """
    Format few-shot examples into a readable string for prompt injection.

    Args:
        few_shot_examples (list[dict]): List of example dicts with keys
            'original_query' and 'new_queries'.
        label (str): Section label for the generated queries. Pass a
            strategy-specific label (e.g., "Sub-Queries", "Step-Back Queries",
            "Paraphrased Queries") to match the calling function's context.
            Defaults to "Expanded Queries".

    Returns:
        str: Formatted string of few-shot examples.
    """
    count = len(few_shot_examples)
    header = "Example" if count == 1 else "Examples"
    few_shot_str = f"{header}:\n\n"
    for example in few_shot_examples:
        few_shot_str += f"Original Query: {example['original_query']}\n"
        few_shot_str += f"{label}:\n"
        for eq in example['new_queries']:
            few_shot_str += f"  - {eq}\n"
        few_shot_str += "\n"
    return few_shot_str


# ---------------------------------------------------------------------------
# Public query-translation functions
# ---------------------------------------------------------------------------

def decompose_query(
    model: BaseChatModel,
    user_query: list[str],
    few_shot_examples: list[dict] | None = None,
) -> list[DecomposedQuery]:
    """
    Break down a complex query into simpler, manageable sub-queries.

    Uses a language model to generate multiple focused sub-questions that
    together cover the full scope of the original query.

    Parameters:
        model (BaseChatModel): LangChain chat model instance.
        user_query (list[str]): Non-empty list of query strings to decompose.
        few_shot_examples (list[dict] | None): Optional override for the prompt
            few-shot examples. Each dict must have keys 'original_query' (str) and
            'new_queries' (list[str]). Defaults to _DEFAULT_DECOMPOSE_EXAMPLES.

    Returns:
        list[DecomposedQuery]: Nested list — one list of DecomposedQuery per input query.

    Raises:
        ValueError: If user_query is not a non-empty list.
        TypeError: If any element of user_query is not a string.
        Exception: Propagated from the LLM or LangChain chain on failure.
    """
    examples = few_shot_examples if few_shot_examples is not None else _DEFAULT_DECOMPOSE_EXAMPLES
    # few_shot_str is supplied as a PromptTemplate variable rather than via f-string
    # interpolation so that braces inside example text cannot corrupt the template.
    few_shot_str = format_few_shot_examples(examples, label="Sub-Queries")

    template = """
        You are an AI assistant specialized in query decomposition.
        Your task is to analyze the input question and generate a set of relevant sub-questions
        that can be answered independently.

        {few_shot_str}

        Please follow these guidelines:
            1. Break down the input query: {question} into distinct sub-queries.
            2. Ensure each sub-query is clear, specific, and focuses on a singular aspect of the original question.
            3. Aim to provide diverse sub-queries that cover various angles of the input question.
        """

    chain = _build_query_chain(model, DecomposedQuery, template)
    return chain.batch(create_input_objects(user_query, few_shot_str=few_shot_str))


def step_back_query(
    model: BaseChatModel,
    user_query: list[str],
    num_queries: int = 3,
    few_shot_examples: list[dict] | None = None,
) -> list[GeneralQuery]:
    """
    Perform step-back prompting to generate higher-level, abstract questions.

    Transforms specific queries into broader conceptual questions that reveal
    underlying principles, enabling retrieval of foundational context.

    The step-back method is a problem-solving and information retrieval technique
    that focuses on generating more abstract or higher-level questions derived from
    the original query. This approach, known as "stepback prompting," emphasizes
    the understanding of broader contexts and underlying concepts.

    Parameters:
        model (BaseChatModel): LangChain chat model instance.
        user_query (list[str]): Non-empty list of query strings to abstract.
        num_queries (int): Number of step-back sub-queries to generate per input.
            Defaults to 3.
        few_shot_examples (list[dict] | None): Optional override for the prompt
            few-shot examples. Defaults to _DEFAULT_STEP_BACK_EXAMPLES.

    Returns:
        list[GeneralQuery]: Nested list — one list of GeneralQuery per input query.

    Raises:
        ValueError: If user_query is not a non-empty list.
        TypeError: If any element of user_query is not a string.
        Exception: Propagated from the LLM or LangChain chain on failure.
    """
    examples = few_shot_examples if few_shot_examples is not None else _DEFAULT_STEP_BACK_EXAMPLES
    few_shot_str = format_few_shot_examples(examples, label="Step-Back Queries")

    # num_queries is an int and safe to bake into the template via f-string.
    # {few_shot_str} and {question} are PromptTemplate variables — escaped as
    # double-braces so the f-string leaves them as literal {placeholders}.
    template = f"""
        You are an advanced cognitive assistant specializing in conceptual query decomposition
        through the "step-back" reasoning technique. Your primary objective is to transform
        specific, narrow queries into broader, more conceptual sub-queries that reveal
        fundamental underlying principles and contexts.

        {{few_shot_str}}

        Step-Back Query Generation Principles:
        1. Conceptual Abstraction
        - Move from specific details to fundamental concepts
        - Identify core principles behind the original query
        - Explore broader intellectual and theoretical frameworks

        2. Multi-Dimensional Analysis
        - Break down the query into distinct conceptual dimensions
        - Uncover hidden relationships and broader contexts
        - Generate sub-queries that provide deeper intellectual insights

        3. Systematic Decomposition Guidelines
        - Original Query: {{question}}
        - Generate exactly {num_queries} step-back sub-queries
        - Each sub-query must:
            * Be more abstract than the original query
            * Reveal underlying theoretical foundations
            * Provide a different perspective on the topic
        """

    chain = _build_query_chain(model, GeneralQuery, template)
    return chain.batch(create_input_objects(user_query, few_shot_str=few_shot_str))


def expand_query(
    model: BaseChatModel,
    user_query: list[str],
    few_shot_examples: list[dict] | None = None,
) -> list[ParaphrasedQuery]:
    """
    Expand a user query into multiple semantically rich paraphrased queries.

    Uses a language model to generate varied phrasings of the original question,
    improving retrieval recall by covering synonyms and alternative formulations.

    Args:
        model (BaseChatModel): LangChain chat model instance.
        user_query (list[str]): Non-empty list of query strings to expand.
        few_shot_examples (list[dict] | None): Optional override for the prompt
            few-shot examples. Defaults to _DEFAULT_EXPAND_EXAMPLES.

    Returns:
        list[ParaphrasedQuery]: Nested list — one list of ParaphrasedQuery per input query.

    Raises:
        ValueError: If user_query is not a non-empty list.
        TypeError: If any element of user_query is not a string.
        Exception: Propagated from the LLM or LangChain chain on failure.
    """
    examples = few_shot_examples if few_shot_examples is not None else _DEFAULT_EXPAND_EXAMPLES
    few_shot_str = format_few_shot_examples(examples, label="Paraphrased Queries")

    template = """
        You are an advanced query expansion specialist designed to generate comprehensive
        and convert user questions into database queries.
        Perform a semantically rich query expansion.
        If there are multiple common ways of phrasing a user question
        or common synonyms for key words in the question, make sure to return multiple versions
        of the query with the different phrasings.
        If there are acronyms or words you are not familiar with, do not try to rephrase them.

        {few_shot_str}

        Return paraphrased versions of the question {question}."""

    chain = _build_query_chain(model, ParaphrasedQuery, template)
    return chain.batch(create_input_objects(user_query, few_shot_str=few_shot_str))


def read_user_queries_from_excel(
    path,
    query_col_index=None,
):
    """
    Read user queries from an Excel file, one list of strings per sheet.

    Parameters:
    - path (str): Path to the Excel file. Queries must be in a single column
      per sheet.
    - query_col_index (list[int] | None): Column index per sheet indicating
      where queries are stored. Must have exactly one entry per sheet.
      Defaults to column 0 for every sheet when None.

    Returns:
    - A tuple containing:
        - queries_list (list[list[str]]): One list of query strings per sheet.
        - sheet_names (list[str]): The names of the sheets in order.

    Raises:
        ValueError: If query_col_index length does not match the number of sheets.
        Exception: If the file cannot be read.
    """

    try:
        sheets_dict = pd.read_excel(path, sheet_name=None)
        dataframes_list = list(sheets_dict.values())
        sheet_names = list(sheets_dict.keys())

        if query_col_index is None:
            query_col_index = [0] * len(dataframes_list)
        elif len(query_col_index) != len(dataframes_list):
            raise ValueError(
                f"query_col_index has {len(query_col_index)} entries but the workbook "
                f"has {len(dataframes_list)} sheets. Provide one index per sheet."
            )

        queries_list = [
            df.iloc[:, query_col_index[i]].tolist()
            for i, df in enumerate(dataframes_list)
        ]

        return queries_list, sheet_names

    except Exception as e:
        raise Exception("An error occurred while reading queries from excel file: " + str(e))
