# 遍歷所有 <a> 標籤，並獲取其文本內容和 href 屬性
for a_tag in all_a_tags:
    print(a_tag.text, a_tag['href'])
