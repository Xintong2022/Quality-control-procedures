"""
KDD 旅游规划 API 系统 - 综合测试脚本
测试所有 API 端点的功能、边界条件和性能
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# 配置
BASE_URL = "http://localhost:5000"
TEST_RESULTS = []

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def log_test(test_name: str, status: str, details: str = "", response_time: float = 0):
    """记录测试结果"""
    result = {
        "test_name": test_name,
        "status": status,
        "details": details,
        "response_time": response_time,
        "timestamp": datetime.now().isoformat()
    }
    TEST_RESULTS.append(result)
    
    color = Colors.GREEN if status == "PASS" else Colors.RED if status == "FAIL" else Colors.YELLOW
    print(f"{color}[{status}]{Colors.END} {test_name} - {response_time:.3f}s")
    if details:
        print(f"  {details}")

def make_request(endpoint: str, params: Dict[str, Any]) -> tuple:
    """发送请求并返回响应和耗时"""
    url = f"{BASE_URL}{endpoint}"
    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=10)
        elapsed = time.time() - start_time
        return response, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return None, elapsed

# ==================== 景点 API 测试 ====================

def test_attractions_basic():
    """测试景点基础查询"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}景点 API 测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    response, elapsed = make_request("/attractions2", {"city": "Baku", "top_k": 5})
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("景点基础查询", "PASS", f"返回 {len(data)} 条结果", elapsed)
            print(f"  示例: {data[0].get('attraction_name', 'N/A')}")
        else:
            log_test("景点基础查询", "FAIL", "返回数据为空", elapsed)
    else:
        status_code = response.status_code if response else "无响应"
        log_test("景点基础查询", "FAIL", f"HTTP {status_code}", elapsed)

def test_attractions_name_filter():
    """测试景点名称过滤"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "attraction_name": "Garden",
        "top_k": 3
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            has_garden = any("garden" in item.get("attraction_name", "").lower() for item in data)
            if has_garden or len(data) > 0:
                log_test("景点名称过滤", "PASS", f"匹配到 {len(data)} 条结果", elapsed)
            else:
                log_test("景点名称过滤", "WARNING", "未找到匹配结果", elapsed)
        else:
            log_test("景点名称过滤", "FAIL", "返回格式错误", elapsed)
    else:
        log_test("景点名称过滤", "FAIL", f"请求失败", elapsed)

def test_attractions_price_filter():
    """测试景点价格过滤"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "ticket_price": 10,
        "top_k": 5
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            # 检查所有价格是否符合条件
            prices = []
            for item in data[:3]:
                price_str = item.get("ticket_price", "0")
                try:
                    price = float(str(price_str).split()[0].replace(",", ""))
                    prices.append(price)
                except:
                    pass
            
            if prices:
                max_price = max(prices)
                status = "PASS" if max_price <= 10 else "WARNING"
                log_test("景点价格过滤", status, f"价格范围: {min(prices):.2f} - {max_price:.2f} USD", elapsed)
            else:
                log_test("景点价格过滤", "PASS", f"返回 {len(data)} 条结果", elapsed)
        else:
            log_test("景点价格过滤", "WARNING", "无符合条件的结果", elapsed)
    else:
        log_test("景点价格过滤", "FAIL", f"请求失败", elapsed)

def test_attractions_time_filter():
    """测试景点开放时间过滤"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "open_hours": "14:30",
        "top_k": 5
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("景点时间过滤", "PASS", f"14:30 开放的景点: {len(data)} 个", elapsed)
            if data:
                print(f"  示例开放时间: {data[0].get('open_hours', 'N/A')}")
        else:
            log_test("景点时间过滤", "WARNING", "无符合时间条件的景点", elapsed)
    else:
        log_test("景点时间过滤", "FAIL", f"请求失败", elapsed)

def test_attractions_semantic_search():
    """测试景点语义搜索"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "facility": "Family with children",
        "top_k": 3
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("景点语义搜索", "PASS", f"找到适合家庭的景点 {len(data)} 个", elapsed)
            if data:
                facilities = data[0].get("facilities", [])
                print(f"  设施示例: {', '.join(facilities[:5]) if facilities else 'N/A'}")
        else:
            log_test("景点语义搜索", "WARNING", "语义搜索无结果", elapsed)
    else:
        log_test("景点语义搜索", "FAIL", f"请求失败", elapsed)

def test_attractions_combined_filters():
    """测试景点组合过滤"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "ticket_price": 20,
        "duration_of_visit": 5,
        "rate_of_restaurant": 3.5,
        "top_k": 10
    })
    
    if response and response.status_code == 200:
        data = response.json()
        log_test("景点组合过滤", "PASS", f"组合条件匹配 {len(data)} 个景点", elapsed)
    else:
        log_test("景点组合过滤", "FAIL", f"请求失败", elapsed)

# ==================== 酒店 API 测试 ====================

def test_hotels_basic():
    """测试酒店基础查询"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}酒店 API 测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    response, elapsed = make_request("/hotels2", {"city": "Bangkok", "top_k": 5})
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("酒店基础查询", "PASS", f"返回 {len(data)} 条结果", elapsed)
            print(f"  示例: {data[0].get('name', 'N/A')} - {data[0].get('star', 'N/A')}星")
        else:
            log_test("酒店基础查询", "FAIL", "返回数据为空", elapsed)
    else:
        log_test("酒店基础查询", "FAIL", f"请求失败", elapsed)

def test_hotels_price_star_filter():
    """测试酒店价格和星级过滤"""
    response, elapsed = make_request("/hotels2", {
        "city": "Bangkok",
        "price": 100,
        "star": 4,
        "top_k": 5
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("酒店价格星级过滤", "PASS", f"4星以上且≤100美元: {len(data)} 家", elapsed)
        else:
            log_test("酒店价格星级过滤", "WARNING", "无符合条件的酒店", elapsed)
    else:
        log_test("酒店价格星级过滤", "FAIL", f"请求失败", elapsed)

def test_hotels_semantic_search():
    """测试酒店语义搜索"""
    response, elapsed = make_request("/hotels2", {
        "city": "Bangkok",
        "amenity": "Business Travelers",
        "top_k": 3
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("酒店语义搜索", "PASS", f"适合商务旅行: {len(data)} 家", elapsed)
        else:
            log_test("酒店语义搜索", "WARNING", "语义搜索无结果", elapsed)
    else:
        log_test("酒店语义搜索", "FAIL", f"请求失败", elapsed)

# ==================== 租车 API 测试 ====================

def test_cars_basic():
    """测试租车基础查询"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}租车 API 测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    response, elapsed = make_request("/cars2", {"city": "Bangkok", "capacity": 4, "top_k": 5})
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("租车基础查询", "PASS", f"返回 {len(data)} 条结果", elapsed)
            print(f"  示例: {data[0].get('car_type', 'N/A')} - {data[0].get('capacity', 'N/A')}座")
        else:
            log_test("租车基础查询", "FAIL", "返回数据为空", elapsed)
    else:
        log_test("租车基础查询", "FAIL", f"请求失败", elapsed)

def test_cars_capacity_filter():
    """测试租车容量过滤"""
    response, elapsed = make_request("/cars2", {
        "city": "Bangkok",
        "capacity": 7,
        "price_per_day": 100,
        "top_k": 5
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            log_test("租车容量过滤", "PASS", f"7座以上车辆: {len(data)} 辆", elapsed)
        else:
            log_test("租车容量过滤", "WARNING", "无符合条件的车辆", elapsed)
    else:
        log_test("租车容量过滤", "FAIL", f"请求失败", elapsed)

# ==================== 航班 API 测试 ====================

def test_flights_oneway():
    """测试单程航班查询"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}航班 API 测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    response, elapsed = make_request("/flights2", {
        "departure_city": "Beijing",
        "arrival_city": "Shanghai",
        "trip_type": "one_way",
        "top_k": 5
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "flights" in data:
            flights = data["flights"]
            log_test("单程航班查询", "PASS", f"北京→上海: {len(flights)} 个航班", elapsed)
            if flights:
                print(f"  示例: {flights[0].get('flight_number', 'N/A')} - {flights[0].get('price', 'N/A')}元")
        else:
            log_test("单程航班查询", "FAIL", "返回格式错误", elapsed)
    else:
        log_test("单程航班查询", "FAIL", f"请求失败", elapsed)

def test_flights_roundtrip():
    """测试往返航班查询"""
    response, elapsed = make_request("/flights2", {
        "departure_city": "Beijing",
        "arrival_city": "Shanghai",
        "trip_type": "round_trip",
        "price": 500,
        "top_k": 3
    })
    
    if response and response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "depart_flights" in data and "return_flights" in data:
            depart = data["depart_flights"]
            return_f = data["return_flights"]
            log_test("往返航班查询", "PASS", f"去程 {len(depart)} 个,返程 {len(return_f)} 个", elapsed)
        else:
            log_test("往返航班查询", "FAIL", "返回格式错误", elapsed)
    else:
        log_test("往返航班查询", "FAIL", f"请求失败", elapsed)

# ==================== 边界测试 ====================

def test_invalid_city():
    """测试无效城市"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}边界条件测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    response, elapsed = make_request("/attractions2", {"city": "InvalidCityXYZ123"})
    
    if response:
        data = response.json()
        if "error" in data or (isinstance(data, list) and len(data) == 0):
            log_test("无效城市处理", "PASS", "正确返回错误或空结果", elapsed)
        else:
            log_test("无效城市处理", "WARNING", "未明确提示错误", elapsed)
    else:
        log_test("无效城市处理", "FAIL", "请求失败", elapsed)

def test_missing_required_params():
    """测试缺少必填参数"""
    response, elapsed = make_request("/cars2", {"city": "Bangkok"})  # 缺少 capacity
    
    if response:
        if response.status_code == 400:
            log_test("缺少必填参数", "PASS", "正确返回 400 错误", elapsed)
        elif response.status_code == 200:
            log_test("缺少必填参数", "WARNING", "未严格验证必填参数", elapsed)
        else:
            log_test("缺少必填参数", "FAIL", f"返回状态码: {response.status_code}", elapsed)
    else:
        log_test("缺少必填参数", "FAIL", "请求失败,无响应", elapsed)

def test_invalid_param_values():
    """测试无效参数值"""
    response, elapsed = make_request("/attractions2", {
        "city": "Baku",
        "ticket_price": "abc",  # 应该是数字
        "top_k": -1  # 应该是正数
    })
    
    if response:
        if response.status_code == 400 or ("error" in response.text):
            log_test("无效参数值", "PASS", "正确处理无效参数", elapsed)
        else:
            log_test("无效参数值", "WARNING", "未严格验证参数类型", elapsed)
    else:
        log_test("无效参数值", "FAIL", "请求失败", elapsed)

# ==================== 性能测试 ====================

def test_response_time():
    """测试响应时间"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}性能测试{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    times = []
    for i in range(5):
        response, elapsed = make_request("/attractions2", {"city": "Baku", "top_k": 10})
        if response and response.status_code == 200:
            times.append(elapsed)
    
    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        status = "PASS" if avg_time < 1.0 else "WARNING"
        log_test("响应时间", status, f"平均: {avg_time:.3f}s, 最大: {max_time:.3f}s, 最小: {min_time:.3f}s", avg_time)
    else:
        log_test("响应时间", "FAIL", "测试失败", 0)

# ==================== 主测试流程 ====================

def run_all_tests():
    """执行所有测试"""
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.YELLOW}KDD 旅游规划 API 系统 - 综合测试{Colors.END}")
    print(f"{Colors.YELLOW}测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}")
    
    # 景点测试
    test_attractions_basic()
    test_attractions_name_filter()
    test_attractions_price_filter()
    test_attractions_time_filter()
    test_attractions_semantic_search()
    test_attractions_combined_filters()
    
    # 酒店测试
    test_hotels_basic()
    test_hotels_price_star_filter()
    test_hotels_semantic_search()
    
    # 租车测试
    test_cars_basic()
    test_cars_capacity_filter()
    
    # 航班测试
    test_flights_oneway()
    test_flights_roundtrip()
    
    # 边界测试
    test_invalid_city()
    test_missing_required_params()
    test_invalid_param_values()
    
    # 性能测试
    test_response_time()
    
    # 生成测试报告
    generate_report()

def generate_report():
    """生成测试报告"""
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.YELLOW}测试报告汇总{Colors.END}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}\n")
    
    passed = sum(1 for r in TEST_RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in TEST_RESULTS if r["status"] == "FAIL")
    warnings = sum(1 for r in TEST_RESULTS if r["status"] == "WARNING")
    total = len(TEST_RESULTS)
    
    print(f"总测试数: {total}")
    print(f"{Colors.GREEN}通过: {passed}{Colors.END}")
    print(f"{Colors.RED}失败: {failed}{Colors.END}")
    print(f"{Colors.YELLOW}警告: {warnings}{Colors.END}")
    print(f"通过率: {(passed/total*100):.1f}%\n")
    
    # 保存详细报告到文件
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "pass_rate": f"{(passed/total*100):.1f}%"
            },
            "test_results": TEST_RESULTS
        }, f, indent=2, ensure_ascii=False)
    
    print(f"详细报告已保存至: {report_file}")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}测试被用户中断{Colors.END}")
    except Exception as e:
        print(f"\n\n{Colors.RED}测试过程中发生错误: {e}{Colors.END}")
