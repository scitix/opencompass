import re
from functools import cache
from typing import List, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# adapted from https://github.com/codelion/optillm/blob/770bf09baa1652c591e4a84cbab3dcffb12ef285/optillm/plugins/readurls_plugin.py
def extract_urls(text: str) -> List[str]:
    # Updated regex pattern to be more precise
    url_pattern = re.compile(r'https?://[^\s\'"]+')

    # Find all matches
    urls = url_pattern.findall(text)

    # Clean up the URLs
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation and quotes
        url = re.sub(r"[,\'\"\)\]]+$", "", url)
        cleaned_urls.append(url)

    return cleaned_urls


@cache
def fetch_webpage_content(url: str, max_length: int = 100000) -> str:
    try:
        headers = {"User-Agent": "optillm/0.0.1 (https://github.com/codelion/optillm)"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Make a soup
        soup = BeautifulSoup(response.content, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text from various elements
        text_elements = []

        # Prioritize content from main content tags
        for tag in ["article", "main", 'div[role="main"]', ".main-content"]:
            content = soup.select_one(tag)
            if content:
                text_elements.extend(
                    content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table"])
                )
                break

        # If no main content found, fall back to all headers, paragraphs, and tables
        if not text_elements:
            text_elements = soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "table"]
            )

        # Process all elements including tables
        content_parts = []

        for element in text_elements:
            if element.name == "table":
                # Process table
                table_content = []

                # Get headers
                headers = element.find_all("th")
                if headers:
                    header_text = " | ".join(
                        header.get_text(strip=True) for header in headers
                    )
                    table_content.append(header_text)

                # Get rows
                for row in element.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if cells:
                        row_text = " | ".join(
                            cell.get_text(strip=True) for cell in cells
                        )
                        table_content.append(row_text)

                # Add table content with proper spacing
                content_parts.append("\n" + "\n".join(table_content) + "\n")
            else:
                # Process regular text elements
                content_parts.append(element.get_text(strip=False))

        # Join all content
        text = " ".join(content_parts)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove footnote superscripts in brackets
        text = re.sub(r"\[.*?\]+", "", text)

        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"


def run(system_prompt, initial_query: str, client=None, model=None) -> Tuple[str, int]:
    urls = extract_urls(initial_query)
    # print(urls)
    modified_query = initial_query

    for url in urls:
        content = fetch_webpage_content(url)
        domain = urlparse(url).netloc
        modified_query = modified_query.replace(
            url, f"{url} [Content from {domain}: {content}]"
        )
    # print(modified_query)
    return modified_query, 0
