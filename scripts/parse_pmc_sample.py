from pathlib import Path
from bs4 import BeautifulSoup


data_dir = Path("data/raw/pmc_oa_comm")
xml_files = list(data_dir.rglob("*.xml"))

print(f"Found XML files: {len(xml_files)}")

if not xml_files:
    raise FileNotFoundError(
        "No XML files found. Please download and extract a PMC oa_comm XML sample first."
    )

sample_file = xml_files[0]
print(f"Sample file: {sample_file}")

with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
    soup = BeautifulSoup(f.read(), "lxml-xml")

title = soup.find("article-title")
abstract = soup.find("abstract")
body = soup.find("body")

title_text = title.get_text(" ", strip=True) if title else "No title found"
abstract_text = abstract.get_text(" ", strip=True) if abstract else "No abstract found"
body_text = body.get_text(" ", strip=True) if body else "No body found"

print("\nTitle:")
print(title_text)

print("\nAbstract preview:")
print(abstract_text[:1000])

print("\nBody preview:")
print(body_text[:1000])
