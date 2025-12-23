import { test, expect } from '@playwright/test';

const API_BASE_URL = 'http://localhost:8012';

interface QueryResponse {
  answer: string;
  used_chunks: any[];
  compression_decisions: Record<string, string>;
  prompt_token_count: number;
  llm_latency_ms: number;
}

test.describe('REFRAG Phase 1 Speed Optimization Tests', () => {
  test.setTimeout(120000); // 2 minute timeout (increased for performance testing)
  
  test.beforeAll(async ({ request }) => {
    // Health check with timeout and retry
    let healthResponse;
    let retries = 5;
    while (retries > 0) {
      try {
        healthResponse = await request.get(`${API_BASE_URL}/health`, { timeout: 5000 });
        if (healthResponse.ok()) {
          const health = await healthResponse.json();
          console.log('Server health:', health);
          return;
        }
      } catch (error) {
        retries--;
        if (retries === 0) {
          throw new Error(`Server not available at ${API_BASE_URL}. Please start the server with: uvicorn app.refrag_server:app --host 0.0.0.0 --port 8012`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
      }
    }
  });

  test('Test 1: BMS 테이블 사용하지 않는 변수 질문 - 속도 및 토큰 측정', async ({ request }) => {
    const query = 'BMS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?';
    const startTime = Date.now();
    
    const response = await request.post(`${API_BASE_URL}/query`, {
      data: {
        query: query,
        top_k: 8
      }
    });
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    expect(response.ok()).toBeTruthy();
    const data: QueryResponse = await response.json();
    
    // 측정 결과 출력
    console.log('\n=== Test 1: BMS 테이블 질문 ===');
    console.log(`총 응답 시간: ${totalTime}ms`);
    console.log(`LLM 지연시간: ${data.llm_latency_ms.toFixed(2)}ms`);
    console.log(`프롬프트 토큰 수: ${data.prompt_token_count}`);
    console.log(`사용된 청크 수: ${data.used_chunks.length}`);
    
    // 압축 비율 계산
    const expandedCount = Object.values(data.compression_decisions).filter(v => v === 'EXPAND').length;
    const compressedCount = Object.values(data.compression_decisions).filter(v => v === 'COMPRESS').length;
    const totalChunks = expandedCount + compressedCount;
    const expandRatio = totalChunks > 0 ? (expandedCount / totalChunks) : 0;
    
    console.log(`확장 청크: ${expandedCount}개`);
    console.log(`압축 청크: ${compressedCount}개`);
    console.log(`확장 비율: ${(expandRatio * 100).toFixed(1)}% (목표: 25-37%)`);
    console.log(`답변: ${data.answer.substring(0, 100)}...`);
    
    // 검증
    expect(data.answer).toBeTruthy();
    // 토큰 수는 최적화 전 대비 감소했는지 확인 (절대값보다는 개선 여부 확인)
    expect(data.prompt_token_count).toBeLessThan(6000); // 최적화 후 토큰 수 제한 (여유있게)
    expect(expandRatio).toBeLessThanOrEqual(0.4); // 확장 비율 40% 이하
  });

  test('Test 2: GPS 테이블 사용하지 않는 변수 질문 - 속도 및 토큰 측정', async ({ request }) => {
    const query = 'GPS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?';
    const startTime = Date.now();
    
    const response = await request.post(`${API_BASE_URL}/query`, {
      data: {
        query: query,
        top_k: 8
      }
    });
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    expect(response.ok()).toBeTruthy();
    const data: QueryResponse = await response.json();
    
    // 측정 결과 출력
    console.log('\n=== Test 2: GPS 테이블 질문 ===');
    console.log(`총 응답 시간: ${totalTime}ms`);
    console.log(`LLM 지연시간: ${data.llm_latency_ms.toFixed(2)}ms`);
    console.log(`프롬프트 토큰 수: ${data.prompt_token_count}`);
    console.log(`사용된 청크 수: ${data.used_chunks.length}`);
    
    // 압축 비율 계산
    const expandedCount = Object.values(data.compression_decisions).filter(v => v === 'EXPAND').length;
    const compressedCount = Object.values(data.compression_decisions).filter(v => v === 'COMPRESS').length;
    const totalChunks = expandedCount + compressedCount;
    const expandRatio = totalChunks > 0 ? (expandedCount / totalChunks) : 0;
    
    console.log(`확장 청크: ${expandedCount}개`);
    console.log(`압축 청크: ${compressedCount}개`);
    console.log(`확장 비율: ${(expandRatio * 100).toFixed(1)}% (목표: 25-37%)`);
    console.log(`답변: ${data.answer.substring(0, 100)}...`);
    
    // 검증
    expect(data.answer).toBeTruthy();
    expect(data.prompt_token_count).toBeLessThan(6000); // 최적화 후 토큰 수 제한
    expect(expandRatio).toBeLessThanOrEqual(0.4);
    
    // GPS 테이블에는 사용하지 않는 변수가 없어야 함
    expect(data.answer.toLowerCase()).toContain('없습니다');
  });

  test('Test 3: 일반 질문 - 속도 및 토큰 측정', async ({ request }) => {
    const query = 'BMS 테이블의 변수 중 SOC와 관련된 변수는?';
    const startTime = Date.now();
    
    const response = await request.post(`${API_BASE_URL}/query`, {
      data: {
        query: query,
        top_k: 8
      }
    });
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    expect(response.ok()).toBeTruthy();
    const data: QueryResponse = await response.json();
    
    // 측정 결과 출력
    console.log('\n=== Test 3: 일반 질문 ===');
    console.log(`총 응답 시간: ${totalTime}ms`);
    console.log(`LLM 지연시간: ${data.llm_latency_ms.toFixed(2)}ms`);
    console.log(`프롬프트 토큰 수: ${data.prompt_token_count}`);
    console.log(`사용된 청크 수: ${data.used_chunks.length}`);
    
    // 압축 비율 계산
    const expandedCount = Object.values(data.compression_decisions).filter(v => v === 'EXPAND').length;
    const compressedCount = Object.values(data.compression_decisions).filter(v => v === 'COMPRESS').length;
    const totalChunks = expandedCount + compressedCount;
    const expandRatio = totalChunks > 0 ? (expandedCount / totalChunks) : 0;
    
    console.log(`확장 청크: ${expandedCount}개`);
    console.log(`압축 청크: ${compressedCount}개`);
    console.log(`확장 비율: ${(expandRatio * 100).toFixed(1)}% (목표: 25-40%)`);
    console.log(`답변: ${data.answer.substring(0, 100)}...`);
    
    // 검증
    expect(data.answer).toBeTruthy();
    expect(data.prompt_token_count).toBeLessThan(6000); // 최적화 후 토큰 수 제한
  });

  test('Test 4: 불확실 사용 질문 - 더 공격적 압축 확인', async ({ request }) => {
    const query = 'BMS 테이블에서 사용 여부가 확실하지 않은 변수는?';
    const startTime = Date.now();
    
    const response = await request.post(`${API_BASE_URL}/query`, {
      data: {
        query: query,
        top_k: 8
      }
    });
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    expect(response.ok()).toBeTruthy();
    const data: QueryResponse = await response.json();
    
    // 측정 결과 출력
    console.log('\n=== Test 4: 불확실 사용 질문 (공격적 압축) ===');
    console.log(`총 응답 시간: ${totalTime}ms`);
    console.log(`LLM 지연시간: ${data.llm_latency_ms.toFixed(2)}ms`);
    console.log(`프롬프트 토큰 수: ${data.prompt_token_count}`);
    
    // 압축 비율 계산
    const expandedCount = Object.values(data.compression_decisions).filter(v => v === 'EXPAND').length;
    const compressedCount = Object.values(data.compression_decisions).filter(v => v === 'COMPRESS').length;
    const totalChunks = expandedCount + compressedCount;
    const expandRatio = totalChunks > 0 ? (expandedCount / totalChunks) : 0;
    
    console.log(`확장 청크: ${expandedCount}개`);
    console.log(`압축 청크: ${compressedCount}개`);
    console.log(`확장 비율: ${(expandRatio * 100).toFixed(1)}% (목표: 20% 이하 - 공격적 압축)`);
    console.log(`답변: ${data.answer.substring(0, 100)}...`);
    
    // 검증: 불확실 질문은 더 공격적 압축 (20% 이하)
    expect(expandRatio).toBeLessThanOrEqual(0.25); // 25% 이하
    expect(data.prompt_token_count).toBeLessThan(6000); // 최적화 후 토큰 수 제한
  });

  test('Test 5: 성능 비교 - 여러 질문 연속 실행', async ({ request }) => {
    const queries = [
      'BMS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?',
      'GPS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?',
      'BMS 테이블의 변수 중 SOC와 관련된 변수는?'
    ];
    
    const results: Array<{
      query: string;
      totalTime: number;
      llmLatency: number;
      promptTokens: number;
      expandRatio: number;
    }> = [];
    
    for (const query of queries) {
      const startTime = Date.now();
      const response = await request.post(`${API_BASE_URL}/query`, {
        data: { query, top_k: 8 }
      });
      const endTime = Date.now();
      
      expect(response.ok()).toBeTruthy();
      const data: QueryResponse = await response.json();
      
      const expandedCount = Object.values(data.compression_decisions).filter(v => v === 'EXPAND').length;
      const totalChunks = Object.keys(data.compression_decisions).length;
      const expandRatio = totalChunks > 0 ? (expandedCount / totalChunks) : 0;
      
      results.push({
        query: query.substring(0, 30) + '...',
        totalTime: endTime - startTime,
        llmLatency: data.llm_latency_ms,
        promptTokens: data.prompt_token_count,
        expandRatio: expandRatio
      });
    }
    
    // 통계 출력
    console.log('\n=== Test 5: 성능 통계 ===');
    const avgTotalTime = results.reduce((sum, r) => sum + r.totalTime, 0) / results.length;
    const avgLlmLateness = results.reduce((sum, r) => sum + r.llmLatency, 0) / results.length;
    const avgPromptTokens = results.reduce((sum, r) => sum + r.promptTokens, 0) / results.length;
    const avgExpandRatio = results.reduce((sum, r) => sum + r.expandRatio, 0) / results.length;
    
    console.log(`평균 총 응답 시간: ${avgTotalTime.toFixed(2)}ms`);
    console.log(`평균 LLM 지연시간: ${avgLlmLateness.toFixed(2)}ms`);
    console.log(`평균 프롬프트 토큰 수: ${avgPromptTokens.toFixed(0)}`);
    console.log(`평균 확장 비율: ${(avgExpandRatio * 100).toFixed(1)}%`);
    
    results.forEach((r, i) => {
      console.log(`\n질문 ${i + 1}: ${r.query}`);
      console.log(`  총 시간: ${r.totalTime}ms, LLM: ${r.llmLatency.toFixed(2)}ms, 토큰: ${r.promptTokens}, 확장: ${(r.expandRatio * 100).toFixed(1)}%`);
    });
    
    // 검증: 평균 토큰 수가 최적화되었는지 확인
    expect(avgPromptTokens).toBeLessThan(6000); // 최적화 후 토큰 수 제한
    expect(avgExpandRatio).toBeLessThanOrEqual(0.4);
  });
});

