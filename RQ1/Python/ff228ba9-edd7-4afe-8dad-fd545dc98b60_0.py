from bs4 import BeautifulSoup

def extract_news(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('h3', class_='gs-c-promo-heading__title gel-pica-bold nw-o-link-split__text').get_text()
    summary = soup.find('p', class_='gs-c-promo-summary gel-long-primer gs-u-mt nw-c-promo-summary').get_text()
    return title, summary

html = '<div><a class="gs-c-promo-heading gs-o-faux-block-link__overlay-link gel-pica-bold nw-o-link-split__anchor" href="/news/uk-politics-65870635"><h3 class="gs-c-promo-heading__title gel-pica-bold nw-o-link-split__text">Country doesn\'t miss Johnson drama, says Shapps</h3></a><p class="gs-c-promo-summary gel-long-primer gs-u-mt nw-c-promo-summary">The energy secretary dismisses Boris Johnson\'s claim that he was the victim of a "witch hunt".</p></div>'
title, summary = extract_news(html)

print('Title:', title)
print('Summary:', summary)
