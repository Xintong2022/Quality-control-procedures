import os
from flask import Flask, request, jsonify, Response
import markdown
from core_api import (
    search_attractions_by_struct_and_group,
    search_hotels_by_struct_and_group,
    search_rental_cars_by_struct_and_group,
    search_flights,
    compute_travel_time,
)


def _loc(args, prefix):
    """Build a location dict from flat query params {prefix}_name/city/type or {prefix}_latitude/longitude."""
    lat, lon = args.get(prefix + "_latitude"), args.get(prefix + "_longitude")
    if lat not in (None, "") and lon not in (None, ""):
        return {"latitude": lat, "longitude": lon}
    return {"name": args.get(prefix + "_name"),
            "city": args.get(prefix + "_city"),
            "type": args.get(prefix + "_type", "attraction")}

app = Flask(__name__)

# ====== 动态查找 api_doc.md 路径 ======
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
# 你可以根据实际位置调整优先查找顺序
API_DOC_CANDIDATES = [
    os.path.join(CUR_DIR, "api_doc.md"),
    os.path.join(PARENT_DIR, "api_doc.md"),
    os.path.join(PARENT_DIR, "data", "api_doc.md"),
]

def find_api_doc():
    for path in API_DOC_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

@app.route("/")
def home():
    doc_path = find_api_doc()
    if doc_path is None:
        return Response("<h1>API Documentation</h1><p>API documentation not found.</p>", mimetype="text/html", status=404)
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    html = markdown.markdown(content)
    return Response(html, mimetype="text/html")

@app.route("/attractions1", methods=["GET"])
def F_list_attractions():
    return jsonify({
        "message": "This API has been deprecated. Please use other attractions endpoints."
    }), 410

from flask import request, jsonify

@app.route("/attractions2", methods=["GET"])
def api_search_attractions2():
    allowed_params = {
        "city",
        "attraction_name",
        "open_hours",
        "ticket_price",
        "max_ticket_price",
        "min_ticket_price",
        "duration_of_visit",
        "max_duration_of_visit",
        "min_duration_of_visit",
        "rate_of_restaurant",
        "min_rate_of_restaurant",
        "max_rate_of_restaurant",
        "facility",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city_name = args.get("city")
    if not city_name or city_name.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400

    result = search_attractions_by_struct_and_group(
        city_name=city_name,
        attraction_name=args.get("attraction_name"),
        open_hours=args.get("open_hours"),
        max_ticket_price=args.get("max_ticket_price"),
        ticket_price=args.get("ticket_price"),
        min_ticket_price=args.get("min_ticket_price"),
        max_duration_of_visit=args.get("max_duration_of_visit"),
        duration_of_visit=args.get("duration_of_visit"),
        min_duration_of_visit=args.get("min_duration_of_visit"),
        min_rate_of_restaurant=args.get("min_rate_of_restaurant"),
        rate_of_restaurant=args.get("rate_of_restaurant"),
        max_rate_of_restaurant=args.get("max_rate_of_restaurant"),
        facilities_group=args.get("facility"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)



@app.route("/hotels1", methods=["GET"])
def F_list_hotels():
    return jsonify({
        "message": "This API has been deprecated. Please use other hotels endpoints."
    }), 410

@app.route("/hotels2", methods=["GET"])
def api_search_hotels2():
    allowed_params = {
        "city",
        "name",
        "price",
        "max_price",
        "min_price",
        "rating",
        "min_rating",
        "max_rating",
        "star",
        "min_star",
        "max_star",
        "rate_of_restaurant",
        "min_rate_of_restaurant",
        "amenity",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city = args.get("city")
    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400

    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        price=args.get("price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        rating=args.get("rating"),
        max_rating=args.get("max_rating"),
        min_star=args.get("min_star"),
        star=args.get("star"),
        max_star=args.get("max_star"),
        min_rate_of_restaurant=args.get("min_rate_of_restaurant"),
        rate_of_restaurant=args.get("rate_of_restaurant"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)




@app.route("/cars1", methods=["GET"])
def F_list_rental_cars():   
    return jsonify({
        "message": "This API has been deprecated. Please use other rental cars endpoints."
    }), 410

@app.route("/cars2", methods=["GET"])
def api_search_rental_cars2():
    allowed_params = {
        "city",
        "capacity",
        "min_capacity",
        "max_capacity",
        "price_per_day",
        "max_price_per_day",
        "min_price_per_day",
        "car_type",
        "extra_service",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city = args.get("city")
    capacity = args.get("capacity")
    # min_capacity is alias for capacity if capacity is present, logic handled in core but we must extract it safely
    # For cars, 'capacity' was required. We should allow 'min_capacity' as well.
    # Current logic: capacity is mandatory param in this func. Let's relax it if min_capacity is there.
    min_capacity = args.get("min_capacity")
    
    # Logic Update: Either capacity OR min_capacity should be present (or both if user mixes)
    # But effectively they mean the same (min requirement).
    effective_capacity = capacity if capacity else min_capacity

    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400
    # Allowed to be None (Optional)
    # if not effective_capacity... logic removed to make it optional consistent with docs

    result = search_rental_cars_by_struct_and_group(
        city_name=city,
        max_price_per_day=args.get("max_price_per_day"),
        price_per_day=args.get("price_per_day"),
        min_price_per_day=args.get("min_price_per_day"),
        car_type=args.get("car_type"),
        min_capacity=args.get("min_capacity"),
        capacity=args.get("capacity"),
        max_capacity=args.get("max_capacity"),
        extra_services_group=args.get("extra_service"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)



@app.route("/flights1", methods=["GET"])
def F_list_flights():
    return jsonify({
        "message": "This API has been deprecated. Please use other flights endpoints."
    }), 410


@app.route("/flights2", methods=["GET"])
def api_search_flight2():
    allowed_params = {
        "departure_city",
        "arrival_city",
        "trip_type",
        "price",
        "max_price",
        "min_price",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    departure_city = args.get("departure_city")
    arrival_city = args.get("arrival_city")
    trip_type = args.get("trip_type")

    if not all([departure_city, arrival_city, trip_type]):
         return jsonify({"error": "Parameters 'departure_city', 'arrival_city', 'trip_type' are required."}), 400

    result = search_flights(
        departure_city=departure_city,
        arrival_city=arrival_city,
        trip_type=trip_type,
        max_price=args.get("max_price"),
        price=args.get("price"),
        min_price=args.get("min_price"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)



# ============ 语义级测试 API (Semantic Testing) ============
# 场景 1: 主 API 返回空结果，需要切换到备选 API
# 场景 2: API 返回错误类型的数据

import random

# 定义一些 "小众城市"，primary API 对这些城市返回空结果
EMPTY_RESULT_CITIES = {
    "Timbuktu", "Ulaanbaatar", "Thimphu", "Vaduz", "Andorra la Vella",
    "San Marino", "Djibouti", "Belmopan", "Nukualofa", "Funafuti",
    "Yaren", "Palikir", "Majuro", "Tarawa", "Apia"
}

@app.route("/hotels_primary", methods=["GET"])
def api_hotels_primary():
    """场景 1: 主 API - 对某些城市返回空结果 (模拟覆盖不全)"""
    args = request.args
    city = args.get("city", "")
    
    # 检查是否是 "小众城市"
    if city in EMPTY_RESULT_CITIES or any(c.lower() in city.lower() for c in EMPTY_RESULT_CITIES):
        # 返回空结果，LLM 应该识别并切换到 backup API
        return jsonify({
            "status": "success",
            "message": f"No hotels found in {city}. Consider using hotels_backup for broader coverage.",
            "data": [],
            "total": 0
        })
    
    # 正常城市：调用真实 API
    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        min_star=args.get("min_star"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 10)),
    )
    return jsonify({"status": "success", "data": result, "total": len(result)})


@app.route("/hotels_backup", methods=["GET"])
def api_hotels_backup():
    """场景 1: 备选 API - 覆盖更广，包括小众城市"""
    args = request.args
    city = args.get("city", "")
    
    # 备选 API 总是返回数据（模拟覆盖更广）
    # 对于小众城市，生成模拟数据
    if city in EMPTY_RESULT_CITIES or any(c.lower() in city.lower() for c in EMPTY_RESULT_CITIES):
        # 生成模拟酒店数据
        mock_hotels = [
            {
                "name": f"Grand Hotel {city}",
                "city": city,
                "price": 120 + random.randint(0, 80),
                "rating": round(3.5 + random.random() * 1.5, 1),
                "star": random.choice([3, 4, 5]),
                "amenities": ["wifi", "parking", "restaurant"]
            },
            {
                "name": f"Budget Inn {city}",
                "city": city,
                "price": 50 + random.randint(0, 30),
                "rating": round(3.0 + random.random(), 1),
                "star": random.choice([2, 3]),
                "amenities": ["wifi"]
            }
        ]
        return jsonify({"status": "success", "data": mock_hotels, "total": len(mock_hotels), "source": "backup"})
    
    # 正常城市：调用真实 API
    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        min_star=args.get("min_star"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 10)),
    )
    return jsonify({"status": "success", "data": result, "total": len(result), "source": "backup"})


@app.route("/search_mixed", methods=["GET"])
def api_search_mixed():
    """场景 2: 混合 API - 随机返回酒店或景点数据 (模拟返回错误类型)
    
    LLM 需要检查返回数据的类型，如果类型不匹配则重新调用正确的 API。
    """
    args = request.args
    city = args.get("city", "")
    requested_type = args.get("type", "hotel")  # hotel 或 attraction
    
    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400
    
    # 50% 概率返回正确类型，50% 概率返回错误类型
    return_wrong_type = random.random() < 0.5
    
    if requested_type == "hotel":
        if return_wrong_type:
            # 错误：返回景点数据
            result = search_attractions_by_struct_and_group(
                city_name=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "hotel",
                "actual_type": "attraction",  # 标记实际返回类型
                "data": result,
                "warning": "Data type mismatch. You requested hotels but received attractions."
            })
        else:
            # 正确：返回酒店数据
            result = search_hotels_by_struct_and_group(
                city=city,
                max_price=args.get("max_price"),
                min_rating=args.get("min_rating"),
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "hotel",
                "actual_type": "hotel",
                "data": result
            })
    else:  # requested_type == "attraction"
        if return_wrong_type:
            # 错误：返回酒店数据
            result = search_hotels_by_struct_and_group(
                city=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "attraction",
                "actual_type": "hotel",  # 标记实际返回类型
                "data": result,
                "warning": "Data type mismatch. You requested attractions but received hotels."
            })
        else:
            # 正确：返回景点数据
            result = search_attractions_by_struct_and_group(
                city_name=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "attraction",
                "actual_type": "attraction",
                "data": result
            })


@app.route("/travel_time", methods=["GET"])
def api_travel_time():
    """Distance + the minimum minutes the plan-feasibility check requires between two locations.
    Each side is given by {from,to}_name + _city + _type (type in hotel/attraction/flight), or by
    {from,to}_latitude + _longitude taken straight from a search result."""
    allowed = {
        "from_name", "from_city", "from_type", "from_latitude", "from_longitude",
        "to_name", "to_city", "to_type", "to_latitude", "to_longitude",
    }
    args = request.args
    bad = [k for k in args.keys() if k not in allowed]
    if bad:
        return jsonify({"error": f"Unsupported parameter(s): {', '.join(bad)}",
                        "allowed_parameters": sorted(allowed)}), 400
    result = compute_travel_time(_loc(args, "from"), _loc(args, "to"))
    return jsonify(result)


if __name__ == "__main__":
    # Fail loudly if a route silently failed to register (the /travel_time-after-app.run() bug).
    _rules = {r.rule for r in app.url_map.iter_rules()}
    for _need in ("/flights2", "/hotels2", "/attractions2", "/cars2", "/travel_time"):
        assert _need in _rules, f"route {_need} not registered — check decorator placement"
    import os as _os
    _port = int(_os.environ.get("TREK_API_PORT", "5001"))
    app.run(host="0.0.0.0", port=_port, debug=False, threaded=True)
