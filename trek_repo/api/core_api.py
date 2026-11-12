# This code is for API calling in Section 3.2

import os
import pandas as pd
import numpy as np
import torch  # Must import before faiss to avoid segfault on Mac ARM
import faiss
import ast
import unicodedata
import glob
import sys
from bedrock_embed import embedding_model  # Titan v2 via Bedrock; replaces the local Qwen model

# travel_time.py (repo root) is the SINGLE source of the B3 travel model, shared with the scorer so
# the compute_travel_time tool reports exactly the gap B3 will require.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from travel_time import haversine_km, min_travel_minutes, travel_mode




CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ATTRACTION_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'v2', 'attraction_data'))
FAC_GROUP_EMB_DIR = os.path.join(ATTRACTION_CSV_DIR, 'facilities_group_embedding')
HOTEL_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'v2', 'hotel_data'))
AMEN_GROUP_EMB_DIR = os.path.join(HOTEL_CSV_DIR, 'amenities_group_embedding')
CAR_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'v2', 'car_data'))
EXTRA_SERVICES_GROUP_EMB_DIR = os.path.join(CAR_CSV_DIR, 'extra_services_group_embedding')
FLIGHT_CSV_PATH = os.path.join(CUR_DIR, 'data', 'v2', 'flight_data', 'flights.csv')

# embedding_model is imported from bedrock_embed (Titan v2). Its .encode(list[str]) returns an
# (n, 1024) float32 array, the same shape the faiss calls below already consume.


# ----------------------------------------------------------------------------
# City resolution
# ----------------------------------------------------------------------------
# Each per-city file used to be found by rebuilding its name from the request:
# `city.strip().title() + "_rental_cars.csv"`. That broke 37 of 375 cities. `.title()` mangles
# lowercase particles ("Ciudad de México" -> "Ciudad De México"); v1 escaped apostrophes to "_"
# ("Xi'an" -> "Xi_an"); and 27 filenames were stored NFD while the city_name column is NFC, so a
# byte-exact match missed. On macOS APFS the last two resolve anyway, which is why this survived
# review -- a reviewer on ext4 would have hit an error dict on 37 cities.
#
# The build now names every file for the NFC city_name it contains, and lookups go through an
# index of what is actually on disk, matched case- and accent-insensitively. Nothing reconstructs
# a filename from a request string any more.
def _fold(name: str) -> str:
    """Case- and accent-insensitive key for matching a requested city to a file on disk."""
    s = unicodedata.normalize('NFKD', str(name)).strip().lower()
    return ''.join(c for c in s if not unicodedata.combining(c))


def _build_city_index(directory: str, suffix: str) -> dict:
    # Two indexes: EXACT NFC name first, folded only as a fallback. 'San Jose' (US) and 'San José'
    # (Costa Rica) are different cities that fold to one key; a folded-only index served one city's
    # data for the other (4,837 km apart) -- the same cross-country collision fixed in data_loader.
    exact, folded = {}, {}
    for path in glob.glob(os.path.join(directory, '*' + suffix)):
        city = unicodedata.normalize('NFC', os.path.basename(path)[:-len(suffix)])
        exact[city] = (city, path)
        folded.setdefault(_fold(city), (city, path))
    return {'exact': exact, 'folded': folded}


_CITY_INDEX = {}


def resolve_city(city_name: str, directory: str, suffix: str):
    """(canonical_city, path) for a requested city, or (None, None) if the KB has no such city."""
    key = (directory, suffix)
    if key not in _CITY_INDEX:
        _CITY_INDEX[key] = _build_city_index(directory, suffix)
    idx = _CITY_INDEX[key]
    nfc = unicodedata.normalize('NFC', str(city_name).strip())
    return idx['exact'].get(nfc) or idx['folded'].get(_fold(city_name), (None, None))


def parse_str_list(val):
    """
    Convert a string-represented list or a real list to a Python list.

    Args:
        val (str or list): Value to parse.
    Returns:
        list: Parsed list, or empty list if parsing fails.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            v = ast.literal_eval(val)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return []

def search_attractions_by_struct_and_group(
    city_name,
    attraction_name=None,
    open_hours=None,
    max_ticket_price=None,
    ticket_price=None, # Legacy alias for max
    min_ticket_price=None,
    max_duration_of_visit=None,
    duration_of_visit=None, # Legacy alias for max
    min_duration_of_visit=None,
    min_rate_of_restaurant=None,
    rate_of_restaurant=None, # Legacy alias for min
    max_rate_of_restaurant=None,
    facilities_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for attractions in a city with structured filters and group-based facility embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (ticket_price, etc.) are supported as aliases.
    """
    # 参数别名处理
    final_max_price = max_ticket_price if max_ticket_price is not None else ticket_price
    final_max_duration = max_duration_of_visit if max_duration_of_visit is not None else duration_of_visit
    final_min_rate = min_rate_of_restaurant if min_rate_of_restaurant is not None else rate_of_restaurant

    city_std, csv_path = resolve_city(city_name, ATTRACTION_CSV_DIR, '_attraction.csv')
    emb_path = (os.path.join(FAC_GROUP_EMB_DIR, f"{city_std}_facilities_group.npy")
                if city_std else '')
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    if attraction_name:
        name_lc = str(attraction_name).lower().strip()
        df = df[df["attraction_name"].str.lower().str.contains(name_lc, na=False)]

    if open_hours:
        def time_to_minutes(time_str):
            h, m = map(int, time_str.strip().split(":"))
            return h * 60 + m
        try:
            param_minute = time_to_minutes(open_hours)
            def check_time(row):
                if pd.isnull(row["open_hours"]): return False
                try:
                    open_str, close_str = row["open_hours"].split('-')
                    open_m = time_to_minutes(open_str)
                    close_m = time_to_minutes(close_str)
                    return open_m <= param_minute <= close_m
                except:
                    return False
            df = df[df.apply(check_time, axis=1)]
        except Exception:
            return {"error": "open_hours should be HH:MM"}

    # 价格提取辅助函数
    def extract_price(s):
        if isinstance(s, str):
            s = s.replace(",", "")
            parts = s.strip().split(" ")[0]
            try:
                return float(parts)
            except:
                return np.nan
        try:
            return float(s)
        except:
            return np.nan

    # 最高价格过滤 (使用统一后的变量)
    if final_max_price is not None:
        try:
            ticket_val = float(final_max_price)
            df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            df = df[df["ticket_price_num"] <= ticket_val]
        except Exception:
            return {"error": "max_ticket_price should be a number"}

    # 最低价格过滤
    if min_ticket_price is not None:
        try:
            min_val = float(min_ticket_price)
            if "ticket_price_num" not in df.columns:
                df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            df = df[df["ticket_price_num"] >= min_val]
        except Exception:
            return {"error": "min_ticket_price should be a number"}

    # 游览时长提取辅助函数
    def get_duration(row):
        try:
            if isinstance(row["duration_of_visit"], str):
                return float(row["duration_of_visit"].split()[0])
            return float(row["duration_of_visit"])
        except:
            return 9999

    # 最长游览时间过滤 (使用统一后的变量)
    if final_max_duration is not None:
        try:
            param_duration = float(final_max_duration)
            df = df[df.apply(lambda row: get_duration(row) <= param_duration, axis=1)]
        except Exception:
            return {"error": "max_duration_of_visit should be a number hour"}

    # 最短游览时间过滤
    if min_duration_of_visit is not None:
        try:
            min_duration = float(min_duration_of_visit)
            df = df[df.apply(lambda row: get_duration(row) >= min_duration, axis=1)]
        except Exception:
            return {"error": "min_duration_of_visit should be a number hour"}

    # 最低餐厅评分过滤 (使用统一后的变量)
    if final_min_rate is not None:
        try:
            param_rate = float(final_min_rate)
            df = df[df["rate_of_restaurant"].astype(float) >= param_rate]
        except Exception:
            return {"error": "min_rate_of_restaurant should be a number"}

    # 最高餐厅评分过滤
    if max_rate_of_restaurant is not None:
        try:
            max_rate = float(max_rate_of_restaurant)
            df = df[df["rate_of_restaurant"].astype(float) <= max_rate]
        except Exception:
            return {"error": "max_rate_of_restaurant should be a number"}

    # 排序逻辑 (新增)
    if sort_by and len(df) > 0:
        sort_field_map = {
            "ticket_price": "ticket_price_num",
            "price": "ticket_price_num",
            "rate_of_restaurant": "rate_of_restaurant",
            "rating": "rate_of_restaurant",
            "duration_of_visit": "duration_of_visit"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            if field == "ticket_price_num" and "ticket_price_num" not in df.columns:
                df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    results = []
    if facilities_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "facilities_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([facilities_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

    final_keys = [
        "attraction_id","city_name","attraction_name","address","longitude","latitude",
        "open_hours","ticket_price","overview","facilities","type",
        "duration_of_visit","rate_of_restaurant"
    ]

    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "facilities" in r:
            r["facilities"] = parse_str_list(r["facilities"])
    return results




def search_hotels_by_struct_and_group(
    city,
    name=None,
    max_price=None,
    price=None, # Legacy alias for max
    min_price=None,
    min_rating=None,
    rating=None, # Legacy alias for min
    max_rating=None,
    min_star=None,
    star=None, # Legacy alias for min
    max_star=None,
    min_rate_of_restaurant=None,
    rate_of_restaurant=None, # Legacy alias for min
    amenities_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for hotels in a city with structured filters and group-based amenities embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (price, rating, etc.) are supported as aliases.
    """
    # 参数别名处理
    final_max_price = max_price if max_price is not None else price
    final_min_rating = min_rating if min_rating is not None else rating
    final_min_star = min_star if min_star is not None else star
    final_min_rate = min_rate_of_restaurant if min_rate_of_restaurant is not None else rate_of_restaurant

    city_std, csv_path = resolve_city(city, HOTEL_CSV_DIR, '_hotel.csv')
    emb_path = (os.path.join(AMEN_GROUP_EMB_DIR, f"{city_std}_amenities_group.npy")
                if city_std else '')
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    if name:
        name_lc = str(name).lower().strip()
        df = df[df["name"].str.lower().str.contains(name_lc, na=False)]

    # 最高价格过滤 (使用统一后的变量)
    if final_max_price is not None:
        try:
            param_price = float(final_max_price)
            df = df[df["price"].astype(float) <= param_price]
        except Exception:
            return {"error": "max_price should be a number"}

    # 最低价格过滤
    if min_price is not None:
        try:
            min_val = float(min_price)
            df = df[df["price"].astype(float) >= min_val]
        except Exception:
            return {"error": "min_price should be a number"}

    # 最低评分过滤 (使用统一后的变量)
    if final_min_rating is not None:
        try:
            param_rating = float(final_min_rating)
            df = df[df["rating"].astype(float) >= param_rating]
        except Exception:
            return {"error": "min_rating should be a number"}

    # 最高评分过滤
    if max_rating is not None:
        try:
            max_rat = float(max_rating)
            df = df[df["rating"].astype(float) <= max_rat]
        except Exception:
            return {"error": "max_rating should be a number"}

    # 最低星级过滤 (使用统一后的变量)
    if final_min_star is not None:
        try:
            param_star = float(final_min_star)
            df = df[df["star"].astype(float) >= param_star]
        except Exception:
            return {"error": "min_star should be a number"}

    # 最高星级过滤
    if max_star is not None:
        try:
            max_s = float(max_star)
            df = df[df["star"].astype(float) <= max_s]
        except Exception:
            return {"error": "max_star should be a number"}

    if final_min_rate is not None:
        try:
            param_rate = float(final_min_rate)
            df = df[df["rate_of_restaurant"].astype(float) >= param_rate]
        except Exception:
            return {"error": "min_rate_of_restaurant should be a number"}

    # 排序逻辑 (新增)
    if sort_by and len(df) > 0:
        sort_field_map = {
            "price": "price",
            "rating": "rating",
            "star": "star"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    results = []
    if amenities_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "amenities_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([amenities_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

    final_keys = [
        "hotel_id","city_name","name","address","price","star","rating",
        "rate_of_restaurant","longitude","latitude",
        "about","amenities"
    ]
    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "amenities" in r:
            r["amenities"] = parse_str_list(r["amenities"])
    return results

def search_rental_cars_by_struct_and_group(
    city_name,
    max_price_per_day=None,
    price_per_day=None, # Legacy alias for max
    min_price_per_day=None,
    car_type=None,
    min_capacity=None,
    capacity=None, # Legacy alias for min
    max_capacity=None,
    extra_services_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for rental cars in a city with structured filters and group-based extra services embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (price_per_day, etc.) are supported as aliases.
    """
    # 参数别名处理
    final_max_price = max_price_per_day if max_price_per_day is not None else price_per_day
    final_min_capacity = min_capacity if min_capacity is not None else capacity

    city_std, csv_path = resolve_city(city_name, CAR_CSV_DIR, '_rental_cars.csv')
    emb_path = (os.path.join(EXTRA_SERVICES_GROUP_EMB_DIR, f"{city_std}_extra_services_group.npy")
                if city_std else '')
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    # 最高价格过滤 (使用统一后的变量)
    if final_max_price is not None:
        try:
            price_val = float(final_max_price)
            df = df[df["price_per_day"].astype(float) <= price_val]
        except Exception:
            return {"error": "max_price_per_day should be a number"}

    # 最低价格过滤
    if min_price_per_day is not None:
        try:
            min_val = float(min_price_per_day)
            df = df[df["price_per_day"].astype(float) >= min_val]
        except Exception:
            return {"error": "min_price_per_day should be a number"}

    if car_type:
        ct = str(car_type).lower().strip()
        df = df[df["car_type"].str.lower().str.contains(ct, na=False)]

    # 最低容量过滤 (使用统一后的变量)
    if final_min_capacity is not None:
        try:
            cap_val = int(final_min_capacity)
            df = df[df["capacity"].astype(int) >= cap_val]
        except Exception:
            return {"error": "min_capacity should be an integer"}

    # 最高容量过滤
    if max_capacity is not None:
        try:
            max_cap = int(max_capacity)
            df = df[df["capacity"].astype(int) <= max_cap]
        except Exception:
            return {"error": "max_capacity should be an integer"}

    # 排序逻辑 (新增)
    if sort_by and len(df) > 0:
        sort_field_map = {
            "price_per_day": "price_per_day",
            "price": "price_per_day",
            "capacity": "capacity"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    if extra_services_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "extra_services_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([extra_services_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

   
    final_keys = [
        "car_id","city_name", "price_per_day", "pickup_location", "car_type", "capacity",
        "extra_services"
    ]
    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "extra_services" in r:
            r["extra_services"] = parse_str_list(r["extra_services"])
    return results

def search_flights(
    departure_city,
    arrival_city,
    trip_type,
    max_price=None,
    price=None, # Legacy alias for max
    min_price=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for flights between two cities with support for one-way and round-trip, filtered by price.
    Arguments accept explicit min/max prefixes. Legacy arguments (price) are supported as aliases.
    """
    # 参数别名处理
    final_max_price = max_price if max_price is not None else price

    dep_city_std = departure_city.strip().title()
    arr_city_std = arrival_city.strip().title()
    
    csv_path = FLIGHT_CSV_PATH
    if not os.path.exists(csv_path):
        return {"error": "flights.csv not found"}
    df = pd.read_csv(csv_path)

    columns_to_keep = [
        "flight_id", "departure_city", "arrival_city", "departure_airport_name", "arrival_airport_name",
        "departure_time", "arrival_time", "flight_number", "price", 
        "departure_airport_latitude", "departure_airport_longitude",
        "arrival_airport_latitude", "arrival_airport_longitude"
    ]

    def apply_filters_and_sort(dataframe):
        # 最高价格过滤 (使用统一变量)
        if final_max_price is not None:
            try:
                p = float(final_max_price)
                dataframe = dataframe[dataframe["price"].astype(float) <= p]
            except:
                pass 
        # 最低价格过滤
        if min_price is not None:
            try:
                min_p = float(min_price)
                dataframe = dataframe[dataframe["price"].astype(float) >= min_p]
            except:
                pass

        # 排序逻辑
        if sort_by and len(dataframe) > 0:
            field = sort_by
            if sort_by == "departure_time":
                field = "departure_time"
            elif sort_by == "price":
                field = "price"
            
            ascending = (sort_order.lower() == "asc")
            dataframe = dataframe.sort_values(by=field, ascending=ascending)
        
        return dataframe, None

  
    if trip_type == "one_way":
        out_flights = df[
            (df["departure_city"].str.lower().str.strip() == dep_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == arr_city_std.lower())
        ]
        
        out_flights, error = apply_filters_and_sort(out_flights)
        if error:
            return error
            
        result = (
            out_flights[columns_to_keep]
            .head(top_k)
            .to_dict(orient="records")
        )
        return {"flights": result}

 
    elif trip_type == "round_trip":
        out_flights = df[
            (df["departure_city"].str.lower().str.strip() == dep_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == arr_city_std.lower())
        ]
        ret_flights = df[
            (df["departure_city"].str.lower().str.strip() == arr_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == dep_city_std.lower())
        ]
        
        out_flights, error = apply_filters_and_sort(out_flights)
        if error:
            return error
        ret_flights, error = apply_filters_and_sort(ret_flights)
        if error:
            return error
            
        result = {
            "depart_flights": (
                out_flights[columns_to_keep]
                .head(top_k)
                .to_dict(orient="records")
            ),
            "return_flights": (
                ret_flights[columns_to_keep]
                .head(top_k)
                .to_dict(orient="records")
            ),
        }
        return result

    else:
        return {"error": "trip_type should be one_way or round_trip"}


# ----------------------------------------------------------------------------
# compute_travel_time — exposes the SCORER's B3 travel model to the agent so
# spatio-temporal feasibility becomes a capability it can plan for, not a hidden
# rule it is scored against. Same haversine + min-travel-time as scoring.py's B3.
# ----------------------------------------------------------------------------
_FLIGHT_POINTS = None


def _flight_points():
    """Flight/airport coordinate table, read ONCE and cached with pre-folded lookup keys.

    This used to re-read flights.csv on every call. The pilot issued 1,090 compute_travel_time calls
    over 220 queries, so the full 800 x 14 run would re-parse the file on the order of 10^5 times —
    a needless bottleneck on a single-process Flask server that has to survive a 12-hour run.
    """
    global _FLIGHT_POINTS
    if _FLIGHT_POINTS is None:
        cols = ['flight_number', 'departure_airport_name', 'arrival_airport_name',
                'departure_airport_iata_code', 'arrival_airport_iata_code',
                'departure_city', 'arrival_city',
                'departure_airport_latitude', 'departure_airport_longitude',
                'arrival_airport_latitude', 'arrival_airport_longitude']
        df = pd.read_csv(FLIGHT_CSV_PATH, keep_default_na=False, usecols=cols)
        for c in ('flight_number', 'departure_airport_name', 'arrival_airport_name',
                  'departure_airport_iata_code', 'arrival_airport_iata_code'):
            df[c + '_key'] = df[c].astype(str).str.strip().str.lower()
        for c in ('departure_city', 'arrival_city'):
            df[c + '_key'] = df[c].map(_fold)
        _FLIGHT_POINTS = df
    return _FLIGHT_POINTS


def _point_coords(loc, side='to'):
    """Resolve a location to (lat, lon). `loc` is {latitude, longitude} or {name, city, type}
    with type in {hotel, attraction, flight}. A flight resolves to its ARRIVAL airport, matching
    how B3 locates a flight event."""
    if not isinstance(loc, dict):
        return None
    lat, lon = loc.get('latitude'), loc.get('longitude')
    if lat not in (None, '') and lon not in (None, ''):
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    name = loc.get('name')
    typ = str(loc.get('type', 'attraction')).strip().lower()

    # `type='flight'` + a city and no name means "the airport serving this city" — a well-defined
    # point in this KB (flights.csv carries departure_city/arrival_city alongside each airport's
    # coordinates) and the second most common way agents asked for an airport in the pilot, after
    # the airport's own name. Returning None here charged the model's spatiotemporal score for a
    # question our data could answer.
    if not name and typ in ('flight', 'airport') and loc.get('city'):
        ckey = _fold(loc.get('city'))
        fdf = _flight_points()
        for side in ('arrival', 'departure'):
            m = fdf[fdf[side + '_city_key'] == ckey]
            if not m.empty:
                r = m.iloc[0]
                try:
                    return float(r[side + '_airport_latitude']), float(r[side + '_airport_longitude'])
                except (TypeError, ValueError):
                    return None
        return None

    if not name:
        return None
    if typ in ('flight', 'airport'):
        # Accept an AIRPORT NAME or IATA code, not just a flight number. Planning "airport -> hotel"
        # is the single most natural use of this tool, and agents overwhelmingly reach for the
        # airport name they just read out of a flight search result ("Zurich Airport",
        # "Phuket International Airport"). Matching only `flight_number` made 46% of all failed
        # compute_travel_time calls in the pilot — 40.8% of calls failed overall — which charged the
        # model's B3 (spatiotemporal) score for a gap in OUR tool.
        key = str(name).strip().lower()
        fdf = _flight_points()
        for col in ('flight_number', 'departure_airport_name', 'arrival_airport_name',
                    'departure_airport_iata_code', 'arrival_airport_iata_code'):
            m = fdf[fdf[col + '_key'] == key]
            if m.empty:
                continue
            r = m.iloc[0]
            # A flight resolves to its ARRIVAL airport (matching how B3 locates a flight event); a
            # named airport resolves to THAT airport wherever it appears.
            # An explicitly NAMED airport resolves to itself. A FLIGHT NUMBER resolves to the
            # endpoint that matches B3's convention, which depends on which side of the hop it is:
            # B3 gives a flight event loc_start = DEPARTURE airport and loc_end = ARRIVAL airport,
            # and measures a hop from e1.loc_end to e2.loc_start. So a flight on the `to_` side is
            # the boarding gate you must REACH (departure), and on the `from_` side it is where you
            # just LANDED (arrival). Resolving a flight number to the arrival airport in both
            # directions made the tool disagree with B3 on 53/53 measured `to_type='flight'` calls,
            # median +207 minutes — while the prompt calls this tool "the exact model the scorer
            # uses". An explicit *_endpoint overrides the default.
            want = str(loc.get('endpoint') or '').strip().lower()
            if want not in ('departure', 'arrival'):
                if col in ('departure_airport_name', 'departure_airport_iata_code'):
                    want = 'departure'
                elif col in ('arrival_airport_name', 'arrival_airport_iata_code'):
                    want = 'arrival'
                else:                                   # matched on flight_number
                    want = 'departure' if side == 'to' else 'arrival'
            try:
                return float(r[want + '_airport_latitude']), float(r[want + '_airport_longitude'])
            except (TypeError, ValueError):
                return None
        return None
    city = loc.get('city')
    if not city:
        return None
    if typ == 'hotel':
        _, path = resolve_city(city, HOTEL_CSV_DIR, '_hotel.csv'); col = 'name'
    else:
        _, path = resolve_city(city, ATTRACTION_CSV_DIR, '_attraction.csv'); col = 'attraction_name'
    if not path:
        return None
    df = pd.read_csv(path)
    m = df[df[col].astype(str).str.strip().str.lower() == str(name).strip().lower()]
    if m.empty:
        return None
    r = m.iloc[0]
    try:
        return float(r['latitude']), float(r['longitude'])
    except (TypeError, ValueError):
        return None


def compute_travel_time(origin, destination):
    """Distance and the MINIMUM minutes the plan-feasibility check (B3) requires between two
    locations. Each of `origin`/`destination` is {latitude, longitude} or {name, city, type}
    (type in {hotel, attraction, flight})."""
    a = _point_coords(origin, side='from')
    b = _point_coords(destination, side='to')
    if a is None or b is None:
        # Say WHICH side failed and WHY. The old blanket message repeated the schema the caller had
        # already read, so an agent that omitted `city` just retried the identical call: 6 straight
        # failures on one pilot query. A specific diagnosis lets it correct itself in one turn.
        def _why(loc, side):   # side is literally "from"/"to", matching _point_coords
            if not isinstance(loc, dict):
                return f"{side}: not an object"
            if _point_coords(loc, side) is not None:
                return None
            if not loc.get("name"):
                return (f"{side}: give either latitude+longitude (copy them straight from a search "
                        f"result) or a name")
            t = str(loc.get("type", "attraction")).strip().lower()
            if t in ("flight", "airport"):
                if not loc.get("name"):
                    return f"{side}: for type 'flight' give a flight number, an airport name, or a `city`"
                return (f"{side}: '{loc.get('name')}' matched no flight_number, airport name or "
                        f"IATA code")
            if not loc.get("city"):
                return f"{side}: type '{t}' also needs `city` (or pass latitude+longitude instead)"
            return f"{side}: no {t} named '{loc.get('name')}' in {loc.get('city')}"
        reasons = [r for r in (_why(origin, "from"), _why(destination, "to")) if r]
        return {"error": "; ".join(reasons) or "could not locate one or both points",
                "hint": "latitude+longitude from a search result always works"}
    d = haversine_km(a[0], a[1], b[0], b[1])
    return {
        "distance_km": round(d, 2),
        "min_travel_minutes": min_travel_minutes(d),
        "mode": travel_mode(d),
    }
