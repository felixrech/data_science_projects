import pycountry
from joblib import Memory
from geopy.location import Location
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from snscrape.modules.twitter import Coordinates

import config


cache = Memory(".cache")

locator = Nominatim(user_agent=config.geocoder_useragent)
_geocode = RateLimiter(locator.geocode, min_delay_seconds=3)
_reverse = RateLimiter(locator.reverse, min_delay_seconds=3)


@cache.cache
def geocode(string) -> Location | None:
    return _geocode(string, language="en", addressdetails=True)


def _country(location: Location | None) -> str | None:
    if location is None or "country_code" not in location.raw["address"].keys():
        return None

    cc_a2 = location.raw["address"]["country_code"]
    country = pycountry.countries.get(alpha_2=cc_a2)
    return country.alpha_3 if country is not None else None


def geocode_country(string) -> str | None:
    ##
    return _country(geocode(string))


def geocode_coordinates(string) -> Coordinates | None:
    location = geocode(string)

    if location is None:
        return None

    return Coordinates(location.longitude, location.latitude)


@cache.cache
def reverse(coordinates: Coordinates | None) -> Location | None:
    if coordinates is None:
        return None

    return _reverse(
        (coordinates.latitude, coordinates.longitude),
        language="en",
        addressdetails=True,
    )


def reverse_country(coordinates: Coordinates) -> str | None:
    ##
    return _country(reverse(coordinates))


def country_name(iso_alpha3: str | None) -> str | None:
    if iso_alpha3 is None:
        return None

    return pycountry.countries.get(alpha_3=iso_alpha3).name
