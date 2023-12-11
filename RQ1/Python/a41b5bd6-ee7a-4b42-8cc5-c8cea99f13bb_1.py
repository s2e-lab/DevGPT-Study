def parse_product(self, response):
    i = ItemLoader(item=Product(), response=response)
    
    i.context['prefix'] = 'EO'
    i.add_value('address', self.address)
    i.add_value('brand', self.name)
    i.add_css('id', 'div.product--price.ordernumber > span')
    i.add_css('sid', 'div.product--price.ordernumber > span')
    i.add_css('title', 'h1.product--title')
    i.add_css('price', 'div.product--price > span')
    i.add_css('size', 'div.product--info > div.product--details > div.product--details--group:nth-child(3) > div.product--details--info:nth-child(2) > div.product--details--value::text')
    i.add_css('time', 'p.delivery--information > span')

    i.add_css('selector', 'ul.breadcrumb--list > li > a > span')

    i.add_value('title_1', 'Deklaration')
    i.add_value('title_2', 'Fütterungsempfehlung')
    i.add_value('title_3', 'Deklaration')
    i.add_value('title_4', 'Fütterungsempfehlung')

    i.add_css('content_1', 'div.product--keywords')
    i.add_css('content_2', 'div.content--description')
    i.add_css('content_3', 'div.product--description')
    i.add_css('content_4', 'div.product--content')

    i.add_css('content_1_html', 'div.product--keywords')
    i.add_css('content_2_html', 'div.content--description')
    i.add_css('content_3_html', 'div.product--description')
    i.add_css('content_4_html', 'div.product--content')

    for img in response.css('div.image-slider--slide > div.image--box > span::attr(data-img-original)'):
        i.add_value('image_urls', img.get())

    item = i.load_item()
    item['size'] = response.css('div.product--info > div.product--details > div.product--details--group:nth-child(3) > div.product--details--info:nth-child(2) > div.product--details--value::text').get()
    yield item
