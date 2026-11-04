// Ramps concurrent virtual users against an OpenAI-compatible chat completions endpoint.
import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter } from 'k6/metrics';

// ============= Configuration =============
const TARGET_URL = __ENV.TARGET_URL || 'http://llm-serve/v1/chat/completions';
const MODEL_NAME = __ENV.MODEL_NAME || 'default';
const MAX_VUS = Math.max(1, parseInt(__ENV.MAX_VUS || '8', 10));
const DURATION_MINUTES = parseFloat(__ENV.DURATION_MINUTES || '10');
const MAX_TOKENS = parseInt(__ENV.MAX_TOKENS || '64', 10);
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || '300s';
const SUMMARY_PATH = __ENV.SUMMARY_PATH || '/tmp/k6-summary.json';

const PROMPTS = [
  'Explain the difference between a process and a thread in two sentences.',
  'List three common causes of high tail latency in a web service.',
  'Summarize what a hash map is and when it is a poor choice.',
  'Write a short definition of backpressure in a message queue.',
  'Describe how a load balancer chooses a backend using least connections.',
  'What is the purpose of a readiness probe in a container orchestrator?',
];

// ============= Metrics =============
const completionTime = new Trend('chat_completion_time', true);
const completionTokens = new Counter('chat_completion_tokens');

// ============= Stages =============
/**
 * Build one stage per virtual user step so the load rises linearly from 1 to MAX_VUS.
 */
function buildStages() {
  const totalSeconds = Math.max(MAX_VUS, Math.round(DURATION_MINUTES * 60));
  const stepSeconds = Math.max(1, Math.floor(totalSeconds / MAX_VUS));
  const stages = [];
  for (let vus = 1; vus <= MAX_VUS; vus += 1) {
    stages.push({ duration: `${stepSeconds}s`, target: vus });
  }
  return stages;
}

export const options = {
  scenarios: {
    chat_ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: buildStages(),
      gracefulRampDown: '30s',
      gracefulStop: '60s',
    },
  },
};

// ============= Test body =============
export default function () {
  const prompt = PROMPTS[(__VU + __ITER) % PROMPTS.length];
  const payload = JSON.stringify({
    model: MODEL_NAME,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: MAX_TOKENS,
    temperature: 0,
    stream: false,
  });
  const response = http.post(TARGET_URL, payload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: REQUEST_TIMEOUT,
  });
  completionTime.add(response.timings.duration);
  check(response, { 'status is 200': (r) => r.status === 200 });
  let tokens = 0;
  if (response.status === 200) {
    try {
      tokens = response.json('usage.completion_tokens') || 0;
    } catch (err) {
      tokens = 0;
    }
    completionTokens.add(tokens);
  }
  // one line per request lets the report split latency before and after scale-out
  console.log(
    `K6_REQ ${JSON.stringify({
      end_ms: Date.now(),
      duration_ms: response.timings.duration,
      status: response.status,
      tokens,
      vus: __VU,
    })}`,
  );
}

// ============= Summary =============
export function handleSummary(data) {
  const body = JSON.stringify(data, null, 2);
  return {
    [SUMMARY_PATH]: body,
    stdout: `K6_SUMMARY_BEGIN\n${JSON.stringify(data)}\nK6_SUMMARY_END\n`,
  };
}
