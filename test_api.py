#!/usr/bin/env python3
"""
Simple test script for Mneme Memory Service API.
"""
import requests
import json
import sys

BASE_URL = "http://localhost:33333"
API_KEY = "test-secret-key"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print(f"✓ Health check passed: {data}")
    return True

def test_create_fact():
    """Test creating a fact."""
    print("Testing create fact...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {"subject": "张三", "predicate": "住在", "object": "上海"}
    response = requests.post(f"{BASE_URL}/facts", headers=headers, json=data)
    assert response.status_code == 201
    result = response.json()
    assert result["subject"] == "张三"
    assert result["predicate"] == "住在"
    assert result["object"] == "上海"
    print(f"✓ Create fact passed: {result['id']}")
    return result["id"]

def test_get_fact(fact_id):
    """Test getting a fact."""
    print("Testing get fact...")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/facts/{fact_id}", headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == fact_id
    print(f"✓ Get fact passed: {result}")
    return True

def test_update_fact(fact_id):
    """Test updating a fact."""
    print("Testing update fact...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {"object": "北京"}
    response = requests.put(f"{BASE_URL}/facts/{fact_id}", headers=headers, json=data)
    print(f"Update response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "北京"
    print(f"✓ Update fact passed: {result}")
    return True

def test_search_semantic():
    """Test semantic search."""
    print("Testing semantic search...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {"query": "张三住在哪里？", "top_k": 3, "mode": "semantic"}
    response = requests.post(f"{BASE_URL}/search", headers=headers, json=data)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Semantic search passed: {len(result.get('facts', []))} facts found")
    return True

def test_search_keyword():
    """Test keyword search."""
    print("Testing keyword search...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {"query": "张三", "top_k": 3, "mode": "keyword"}
    response = requests.post(f"{BASE_URL}/search", headers=headers, json=data)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Keyword search passed: {len(result.get('facts', []))} facts found")
    return True

def test_search_hybrid():
    """Test hybrid search."""
    print("Testing hybrid search...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {"query": "张三住在哪里？", "top_k": 3, "mode": "hybrid"}
    response = requests.post(f"{BASE_URL}/search", headers=headers, json=data)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Hybrid search passed: {len(result.get('facts', []))} facts found")
    return True

def test_batch_create():
    """Test batch create facts."""
    print("Testing batch create facts...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = {
        "facts": [
            {"subject": "李四", "predicate": "住在", "object": "北京"},
            {"subject": "王五", "predicate": "住在", "object": "广州"}
        ]
    }
    response = requests.post(f"{BASE_URL}/facts/batch", headers=headers, json=data)
    assert response.status_code == 201
    result = response.json()
    assert len(result) == 2
    print(f"✓ Batch create passed: {len(result)} facts created")
    return [r["id"] for r in result]

def test_delete_fact(fact_id):
    """Test deleting a fact."""
    print("Testing delete fact...")
    headers = {"X-API-Key": API_KEY}
    response = requests.delete(f"{BASE_URL}/facts/{fact_id}", headers=headers)
    assert response.status_code == 204
    print(f"✓ Delete fact passed: {fact_id}")
    return True

def test_metrics():
    """Test metrics endpoint."""
    print("Testing metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    result = response.json()
    assert "requests_total" in result
    print(f"✓ Metrics passed: {result}")
    return True

def test_export():
    """Test export endpoint."""
    print("Testing export endpoint...")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/export", headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert "facts" in result
    assert "summaries" in result
    print(f"✓ Export passed: {len(result['facts'])} facts, {len(result['summaries'])} summaries")
    return True

def main():
    """Run all tests."""
    print("Starting Mneme Memory Service API tests...")
    print("=" * 50)
    
    try:
        # Test health
        test_health()
        
        # Test CRUD operations
        fact_id = test_create_fact()
        test_get_fact(fact_id)
        test_update_fact(fact_id)
        
        # Test search
        test_search_semantic()
        test_search_keyword()
        test_search_hybrid()
        
        # Test batch operations
        batch_ids = test_batch_create()
        
        # Test metrics
        test_metrics()
        
        # Test export
        test_export()
        
        # Cleanup
        test_delete_fact(fact_id)
        for bid in batch_ids:
            test_delete_fact(bid)
        
        print("=" * 50)
        print("✓ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
