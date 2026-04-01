"""
Java 教程数据采集模块
从 Runoob.com 爬取 Java 教程内容
"""
import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from config import RAW_DATA_DIR

BASE_URL = "https://www.runoob.com"
JAVA_INDEX_URL = f"{BASE_URL}/java/java-tutorial.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_java_tutorial_links() -> list:
    """获取 Java 教程的所有章节链接"""
    print("[爬虫] 正在获取 Java 教程目录...")
    resp = requests.get(JAVA_INDEX_URL, headers=HEADERS, timeout=30)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    links = []
    # Runoob 的侧边栏目录
    sidebar = soup.find("div", {"id": "leftcolumn"}) or soup.find("div", class_="design-left")
    if sidebar:
        for a in sidebar.find_all("a", href=True):
            href = a["href"]
            if "/java/" in href and href.endswith(".html"):
                full_url = href if href.startswith("http") else BASE_URL + href
                title = a.get_text(strip=True)
                if title and full_url not in [l["url"] for l in links]:
                    links.append({"title": title, "url": full_url})

    print(f"[爬虫] 发现 {len(links)} 个 Java 教程页面")
    return links


def extract_page_content(url: str) -> str:
    """提取页面的教学正文内容"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        # Runoob 的文章内容区域
        content_div = soup.find("div", {"id": "content"}) or soup.find("div", class_="article-body")
        if not content_div:
            return ""

        # 移除脚本、样式、广告等
        for tag in content_div.find_all(["script", "style", "ins", "iframe"]):
            tag.decompose()

        # 提取文本，保留代码块
        text_parts = []
        for element in content_div.children:
            if hasattr(element, 'name'):
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    text_parts.append(f"\n## {element.get_text(strip=True)}\n")
                elif element.name == 'pre':
                    code = element.get_text()
                    text_parts.append(f"\n```java\n{code}\n```\n")
                elif element.name in ['p', 'div', 'li', 'td']:
                    text = element.get_text(strip=True)
                    if text:
                        text_parts.append(text)
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li'):
                        text_parts.append(f"- {li.get_text(strip=True)}")
                elif element.name == 'table':
                    rows = element.find_all('tr')
                    for row in rows:
                        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                        if cells:
                            text_parts.append(" | ".join(cells))

        return "\n".join(text_parts)

    except Exception as e:
        print(f"[爬虫错误] 提取 {url} 失败: {e}")
        return ""


def crawl_java_tutorials(max_pages: int = None) -> list:
    """
    爬取 Java 教程全部内容
    返回: [{"title": "章节标题", "url": "链接", "content": "正文内容"}, ...]
    """
    links = get_java_tutorial_links()
    if max_pages:
        links = links[:max_pages]

    tutorials = []
    for idx, link in enumerate(links):
        print(f"[爬虫] [{idx+1}/{len(links)}] 正在爬取: {link['title']}")
        content = extract_page_content(link["url"])
        if content.strip():
            tutorials.append({
                "title": link["title"],
                "url": link["url"],
                "content": content
            })
        time.sleep(1)  # 礼貌爬虫

    # 保存到本地
    output_file = os.path.join(RAW_DATA_DIR, "java_tutorials.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tutorials, f, ensure_ascii=False, indent=2)

    print(f"[爬虫] 完成！共爬取 {len(tutorials)} 篇教程，保存至 {output_file}")
    return tutorials


def load_local_tutorials() -> list:
    """加载本地已爬取的教程数据"""
    filepath = os.path.join(RAW_DATA_DIR, "java_tutorials.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_custom_text(file_path: str) -> list:
    """加载自定义文本文件（支持 .txt, .md）"""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按章节分割
    sections = re.split(r'\n#{1,3}\s+', content)
    result = []
    for i, section in enumerate(sections):
        if section.strip():
            lines = section.strip().split('\n', 1)
            title = lines[0].strip() if lines else f"Section {i+1}"
            body = lines[1].strip() if len(lines) > 1 else section.strip()
            result.append({"title": title, "content": body})

    return result


if __name__ == "__main__":
    tutorials = crawl_java_tutorials(max_pages=5)
    for t in tutorials:
        print(f"\n{'='*50}")
        print(f"标题: {t['title']}")
        print(f"内容预览: {t['content'][:200]}...")
