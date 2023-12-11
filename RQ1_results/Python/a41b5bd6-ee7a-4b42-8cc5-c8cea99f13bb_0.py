from PIMS.spiders.base import BaseSpider
from scrapy.loader import ItemLoader
from PIMS.items import Product
from scrapy import Request


class StaehlerSchopfSpider(BaseSpider):

    name = 'StaehlerSchopf'
    address = 'YOUR_ADDRESS'  # Set your desired address
    allowed_domains = ['dr.staehler-schopf.de']
    start_urls = ['https://dr.staehler-schopf.de/']

    def parse(self, response):
        for item in response.css('div.t3-megamenu--container > div > ul > li > a::attr(href)'):
            yield Request(url=response.urljoin(item.get()), callback=self.parse_category)

    def parse_category(self, response):
        for item in response.css('div.product--title > a::attr(href)'):
            yield Request(url=response.urljoin(item.get()), callback=self.parse_product)

    def parse_product(self, response):
        i = ItemLoader(item=Product(), response=response)

        i.context['prefix'] = 'SS'
        i.add_value('address', self.address)
        i.add_value('brand', self.name)
        i.add_css('id', 'div.product--content > div > span')
        i.add_css('sid', 'div.product--content > div > span')
        i.add_css('title', 'h1.product--title')
        i.add_css('price', 'span.price')
        i.add_css('size', 'div.product--size')
        i.add_css('time', 'div.delivery--info')

        i.add_css('selector', 'ul.breadcrumb--list > li > a > span')

        i.add_value('title_1', 'Zusammenfassung')
        i.add_value('title_2', 'Produktinformationen')

        i.add_css('content_1', 'div.product--summary')
        i.add_css('content_2', 'div.product--description')

        i.add_css('content_1_html', 'div.product--summary')
        i.add_css('content_2_html', 'div.product--description')

        for img in response.css('div.product--images > img::attr(src)'):
            i.add_value('image_urls', response.urljoin(img.get()))

        return i.load_item()
