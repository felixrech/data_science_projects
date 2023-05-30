# Analyzing opinions concerning the Russian invasion of Ukraine using ABSA

Aspect-based sentiment analysis tries to extract the sentiment concerning specific aspects of a text. For example, consider the sentence "I like pizza but hate pineapples." would have an overall mixed/neutral sentiment. But if one focuses on just the aspect "pizza", the author's sentiment is quite positive.

We scrape tweets concerning the Russian invasion of Ukraine (currently just tweets containing "Putin" with at least fifty likes and in the period from 2023-01-01 until 2023-04-30). Then a simple ABSA approach is applied to extract sentiment toward Putin himself. For the results, check out [putin.ipynb](putin.ipynb).

## Warning

You will need to get and enter your own Mapbox token in the [config.py](config.py) file before running the Jupyter notebook yourself!

Everything is still very much in active development. In particular, expect missing or outright wrong documentation of code. The current caching process is also patchwork only and will be switched to a small database backend (using PostGIS so I can get some training using that, too).

Planned developments include:
- Setup and implementation of proper database backend
- Scraping of a longer time period (starting before the beginning of the invasion)
- Extension of aspects/search terms to include, e.g. NATO, Ukraine, etc.

## Attributions

The world geoJSON data comes from [https://geojson-maps.ash.ms/](https://geojson-maps.ash.ms/).