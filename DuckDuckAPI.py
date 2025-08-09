from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


results = DDGS().text("إذا كان القياس متفرعا من الكتاب والسنة والإجماع والعقل، فما الذي كان يجب ألا يذكروه في شروط الاجتهاد؟", region = "ar-sa", max_results=2)
urls = [result['href'] for result in results]  # Lấy danh sách URL
# Hàm thu thập dữ liệu từ một URL
def scrape_page(url):
    try:
        # Gửi yêu cầu HTTP đến URL
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Kiểm tra lỗi HTTP

        # Phân tích HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Lấy tiêu đề trang
        title = soup.find('title').text if soup.find('title') else 'No title'

        # Lấy toàn bộ văn bản từ thẻ <p> (hoặc tùy chỉnh theo cấu trúc trang)
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs])

        return {'url': url, 'title': title, 'content': content}
    except Exception as e:
        print(f"Lỗi khi thu thập {url}: {e}")
        return None

# Thu thập dữ liệu từ tất cả URL
scraped_data = []
for url in urls:
    data = scrape_page(url)
    if data:
        scraped_data.append(data)

# In kết quả
for data in scraped_data:
    print(f"URL: {data['url']}")
    print(f"Tiêu đề: {data['title']}")
    print(f"Nội dung: {data['content']}...")  