# 查找 class 為 'title' 的 <div> 標籤
div_with_class_title = soup.select_one('div.title')

# 查找所有 class 為 'item' 的 <li> 標籤
all_li_with_class_item = soup.select('li.item')
