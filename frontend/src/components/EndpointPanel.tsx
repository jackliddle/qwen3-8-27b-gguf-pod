import { useState } from 'react'
import { getKey, type Deployment } from '../api'
import { CopyButton } from './CopyButton'

type Tab = 'curl' | 'python' | 'env'

export function EndpointPanel({ dep }: { dep: Deployment }) {
  const [reveal, setReveal] = useState(false)
  const [tab, setTab] = useState<Tab>('curl')
  const key = getKey()
  const { base_url, model } = dep.endpoint

  const snippets: Record<Tab, string> = {
    curl: `curl ${base_url}/chat/completions \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello"}]}'`,
    python: `from openai import OpenAI

client = OpenAI(base_url="${base_url}", api_key="${key}")
resp = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)`,
    env: `OPENAI_BASE_URL=${base_url}
OPENAI_API_KEY=${key}
OPENAI_MODEL=${model}`,
  }
  const shown = reveal || !key ? snippets[tab] : snippets[tab].split(key).join('•'.repeat(12))

  return (
    <div className="endpoint">
      <dl className="kv">
        <dt>Base URL</dt>
        <dd>
          <code>{base_url}</code>
          <CopyButton text={base_url} />
        </dd>
        <dt>Model</dt>
        <dd>
          <code>{model}</code>
          <CopyButton text={model} />
        </dd>
        <dt>API key</dt>
        <dd>
          <code>{reveal ? key : '•'.repeat(12)}</code>
          <button className="btn btn-ghost btn-xs" type="button" onClick={() => setReveal(!reveal)}>
            {reveal ? 'Hide' : 'Show'}
          </button>
          <CopyButton text={key} />
        </dd>
      </dl>
      <div className="snippet">
        <div className="tabs" role="tablist">
          {(['curl', 'python', 'env'] as Tab[]).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={`tab ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              type="button"
            >
              {t === 'env' ? '.env' : t}
            </button>
          ))}
          <span className="spacer" />
          <CopyButton text={snippets[tab]} />
        </div>
        <pre>{shown}</pre>
      </div>
    </div>
  )
}
