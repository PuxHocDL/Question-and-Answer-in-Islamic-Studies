from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


results = DDGS().text("إذا كان القياس متفرعا من الكتاب والسنة والإجماع والعقل، فما الذي كان يجب ألا يذكروه في شروط الاجتهاد؟", region = "ar-sa", max_results=2)

print(results)