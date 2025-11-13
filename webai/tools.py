import os
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
# from tavily import TavilyClient


class SearchProvider(Enum):
    """Supported web search providers."""
    TAVILY = "tavily"
    OPENAI = "openai"


@dataclass
class SearchResult:
    """Standardized search result format."""
    title: str
    content: str
    source: str
    raw_data: dict


class WebSearcher:
    """
    Unified web search interface supporting Tavily and OpenAI web search.
    """

    def __init__(
        self,
        provider: SearchProvider = SearchProvider.OPENAI,
        tavily_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        max_results: int = 5,
        include_raw_content: bool = False,
        debug: bool = True,
    ):
        """
        Initialize the web searcher.

        Args:
            provider: Search provider to use (TAVILY or OPENAI)
            tavily_api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_results: Maximum number of results to return
            include_raw_content: Include full page content (Tavily only)
            debug: Print debug information
        """
        self.provider = provider
        self.max_results = max_results
        self.include_raw_content = include_raw_content
        self.debug = debug

        # Always initialize both clients
        self.tavily_tool = None
        self.llm = None
        self.llm_with_search = None

        # Initialize Tavily
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if tavily_key and self.debug:
            print(f"✓ Tavily API key loaded: {tavily_key[:10]}...")
        
        if tavily_key:
            try:
                self.tavily_tool = TavilySearch(
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                )
                if self.debug:
                    print("✓ Tavily tool initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize Tavily: {e}")

        # Initialize OpenAI
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_key and self.debug:
            print(f"✓ OpenAI API key loaded: {openai_key[:10]}...")
        
        if openai_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_key
                )
                if self.debug:
                    print("✓ ChatOpenAI initialized successfully")
                
                # Bind web search tool
                self.llm_with_search = self.llm.bind_tools([
                    {"type": "web_search_preview"}
                ])
                if self.debug:
                    print("✓ Web search tool bound to LLM")
                    
            except Exception as e:
                print(f"✗ Failed to initialize OpenAI: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def search_tavily(
        self,
        query: str,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> list[SearchResult]:
        """
        Search using Tavily API.

        Args:
            query: Search query
            topic: Search topic category (general, news, or finance)

        Returns:
            List of standardized search results
        """
        if not self.tavily_tool:
            raise ValueError("Tavily API key not configured")

        if self.debug:
            print(f"\n[DEBUG] Invoking Tavily with query: {query}, topic: {topic}")

        try:
            # TavilySearch returns the raw response from Tavily API
            response = self.tavily_tool.invoke({"query": query, "topic": topic})
            
            if self.debug:
                print(f"[DEBUG] Tavily response type: {type(response)}")
                print(f"[DEBUG] Tavily response: {response}")
            
            results = []
            
            # Handle response format from TavilySearch
            if isinstance(response, dict):
                for result in response.get("results", []):
                    results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            content=result.get("content", "")[:500],
                            source=result.get("url", ""),
                            raw_data=result,
                        )
                    )
            elif isinstance(response, list):
                for result in response:
                    if isinstance(result, dict):
                        results.append(
                            SearchResult(
                                title=result.get("title", ""),
                                content=result.get("content", "")[:500],
                                source=result.get("url", ""),
                                raw_data=result,
                            )
                        )
            
            if self.debug:
                print(f"[DEBUG] Tavily results found: {len(results)}")
            
            return results[:self.max_results]

        except Exception as e:
            print(f"✗ Tavily search error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Tavily search failed: {str(e)}")

    def search_openai(self, query: str) -> list[SearchResult]:
        """
        Search using OpenAI's native web search.

        Args:
            query: Search query

        Returns:
            List of standardized search results
        """
        if not self.llm_with_search:
            raise ValueError("OpenAI API key not configured")

        if self.debug:
            print(f"\n[DEBUG] Invoking OpenAI with query: {query}")

        try:
            # Invoke the LLM with web search enabled
            response = self.llm_with_search.invoke(
                f"Search the web for: {query}. Return the top {self.max_results} results with sources."
            )

            if self.debug:
                print(f"[DEBUG] Response type: {type(response)}")
                print(f"[DEBUG] Response: {response}")
                if hasattr(response, "content"):
                    print(f"[DEBUG] Response content type: {type(response.content)}")
                    print(f"[DEBUG] Response content: {response.content}")

            results = []
            
            # Parse the response
            if hasattr(response, "content"):
                content = response.content
                
                if self.debug:
                    print(f"[DEBUG] Content length: {len(str(content))}")
                
                # OpenAI returns content as string
                if isinstance(content, str):
                    if self.debug:
                        print(f"[DEBUG] Creating SearchResult from string content")
                    results.append(
                        SearchResult(
                            title=query,
                            content=content[:500],
                            source="openai-search",
                            raw_data={"response": str(response)},
                        )
                    )
                elif isinstance(content, list):
                    if self.debug:
                        print(f"[DEBUG] Content is list with {len(content)} items")
                    for item in content:
                        if isinstance(item, dict):
                            results.append(
                                SearchResult(
                                    title=item.get("title", query),
                                    content=item.get("text", "")[:500],
                                    source=item.get("url", "openai-search"),
                                    raw_data=item,
                                )
                            )
            
            if self.debug:
                print(f"[DEBUG] Total results found: {len(results)}")
            
            return results[:self.max_results]

        except Exception as e:
            print(f"✗ OpenAI search error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"OpenAI search failed: {str(e)}")

    def search(
        self,
        query: str,
        provider: Optional[SearchProvider] = None,
        topic: Literal["general", "news", "finance"] = "general",
        fallback: bool = True,
    ) -> list[SearchResult]:
        """
        Perform web search with optional provider override.

        Args:
            query: Search query
            provider: Override default provider (uses self.provider if None)
            topic: Topic for Tavily search (ignored for OpenAI)
            fallback: Attempt fallback provider if primary fails

        Returns:
            List of search results
        """
        selected_provider = provider or self.provider
        
        if self.debug:
            print(f"\n[DEBUG] Search called with:")
            print(f"  - Query: {query}")
            print(f"  - Provider: {selected_provider}")
            print(f"  - Fallback enabled: {fallback}")

        try:
            if selected_provider == SearchProvider.TAVILY:
                return self.search_tavily(query, topic=topic)
            elif selected_provider == SearchProvider.OPENAI:
                return self.search_openai(query)
            else:
                raise ValueError(f"Unknown provider: {selected_provider}")

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Exception caught: {e}")
            
            if fallback:
                # Try fallback provider
                fallback_provider = (
                    SearchProvider.OPENAI 
                    if selected_provider == SearchProvider.TAVILY 
                    else SearchProvider.TAVILY
                )
                
                if self.debug:
                    print(f"[DEBUG] Trying fallback provider: {fallback_provider}")
                
                try:
                    if fallback_provider == SearchProvider.TAVILY:
                        return self.search_tavily(query, topic=topic)
                    else:
                        return self.search_openai(query)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both searches failed.\n"
                        f"Primary ({selected_provider}): {e}\n"
                        f"Fallback ({fallback_provider}): {fallback_error}"
                    )
            else:
                raise

    def format_results(self, results: list[SearchResult]) -> str:
        """
        Format search results as readable text.
        """
        if self.debug:
            print(f"[DEBUG] Formatting {len(results)} results")
        
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.title}\n"
                f"   Source: {result.source}\n"
                f"   {result.content}\n"
            )
        return "\n".join(formatted)

    def get_content_only(self, results: list[SearchResult]) -> str:
        """
        Extract only the content from search results.
        
        Args:
            results: List of search results
            
        Returns:
            Just the content text, joined by newlines
        """
        return "\n".join([result.content for result in results])

    def get_content_list(self, results: list[SearchResult]) -> list[str]:
        """
        Get content as a list of strings (one per result).
        
        Args:
            results: List of search results
            
        Returns:
            List of content strings
        """
        return [result.content for result in results]

    def format_content_only(self, results: list[SearchResult]) -> str:
        """
        Format results with only content, no titles or sources.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted content with separators
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"--- Result {i} ---")
            formatted.append(result.content)
            formatted.append("")
        
        return "\n".join(formatted)

    def format_minimal(self, results: list[SearchResult]) -> str:
        """
        Format results with title and content only (no source).
        
        Args:
            results: List of search results
            
        Returns:
            Formatted output with titles and content
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result.title}")
            formatted.append(result.content)
            formatted.append("")
        
        return "\n".join(formatted)

    def get_first_content(self, results: list[SearchResult]) -> Optional[str]:
        """
        Get the content from the first result.
        
        Args:
            results: List of search results
            
        Returns:
            Content from first result, or None if no results
        """
        return results[0].content if results else None

    def format_results(self, results: list[SearchResult]) -> str:
        """
        Format search results as readable text (original method with all info).
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.title}\n"
                f"   Source: {result.source}\n"
                f"   {result.content}\n"
            )
        return "\n".join(formatted)
    
    def to_documents(self, results: list[SearchResult]) -> list[Document]:
        """
        Convert search results to LangChain Document objects.
        
        Each result becomes a Document with:
        - page_content: The search result content
        - metadata: Title, source, and raw data
        
        Args:
            results: List of search results
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for result in results:
            doc = Document(
                page_content=result.content,
                metadata={
                    "title": result.title,
                    "source": result.source,
                    "raw_data": result.raw_data,
                }
            )
            documents.append(doc)
        
        return documents



# # Usage examples
# if __name__ == "__main__":
#     from dotenv import load_dotenv
    
#     load_dotenv()

#     # Test with OpenAI
#     print("=" * 60)
#     print("Testing OpenAI Web Search")
#     print("=" * 60)
    
#     try:
#         searcher_openai = WebSearcher(provider=SearchProvider.OPENAI)
#         results_openai = searcher_openai.search("latest Python frameworks 2025")
#         print(searcher_openai.format_results(results_openai))
#     except Exception as e:
#         print(f"Error: {e}")

#     print("\n" + "=" * 60)
#     print("Testing Tavily Web Search (Finance)")
#     print("=" * 60)
    
#     try:
#         searcher_tavily = WebSearcher(
#             provider=SearchProvider.TAVILY,
#             max_results=3,
#             include_raw_content=False,
#         )
#         results_tavily = searcher_tavily.search(
#             "artificial intelligence stocks",
#             topic="finance"
#         )
#         print(searcher_tavily.format_results(results_tavily))
#     except Exception as e:
#         print(f"Error: {e}")

#     print("\n" + "=" * 60)
#     print("Testing Fallback (Tavily → OpenAI)")
#     print("=" * 60)
    
#     try:
#         searcher_auto = WebSearcher(provider=SearchProvider.TAVILY)
#         results_auto = searcher_auto.search(
#             "LangChain documentation",
#             fallback=True
#         )
#         print(searcher_auto.format_results(results_auto))
#     except Exception as e:
#         print(f"Error: {e}")