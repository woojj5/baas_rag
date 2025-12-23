"""Test script for REFRAG API."""
import asyncio
import httpx


async def test_refrag_api():
    """Test REFRAG API endpoint."""
    base_url = "http://localhost:8011"
    
    # Test health
    print("Testing /health...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        print(f"Health status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    
    # Test query
    print("Testing /query...")
    query_data = {
        "query": "BMS 테이블에 존재는 하지만 실제로는 사용되지 않는 변수는?",
        "top_k": 8
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/query",
            json=query_data
        )
        print(f"Query status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer'][:200]}...")
            print(f"Used chunks: {len(data['used_chunks'])}")
            print(f"Compression decisions: {data['compression_decisions']}")
            print(f"Prompt tokens: {data['prompt_token_count']}")
            print(f"LLM latency: {data['llm_latency_ms']:.2f}ms")
        else:
            print(f"Error: {response.text}")


if __name__ == "__main__":
    asyncio.run(test_refrag_api())

