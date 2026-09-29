import { useState } from 'react'
import { getKey, type Deployment } from '../api'
import { CopyButton } from './CopyButton'

type Tab = 'curl' | 'python' | 'env' | 'job'

export function EndpointPanel({ dep }: { dep: Deployment }) {
  const [reveal, setReveal] = useState(false)
  const [tab, setTab] = useState<Tab>('curl')
  const key = getKey()
  const { base_url, model } = dep.endpoint

  const env = `OPENAI_BASE_URL=${base_url}
OPENAI_API_KEY=${key}
OPENAI_MODEL=${model}`
  const root = base_url.replace(/\/v1$/, '')
  const edit = dep.capabilities.includes('img2img') && !dep.capabilities.includes('txt2img')
  const snippets: Partial<Record<Tab, string>> =
    dep.kind !== 'image'
      ? {
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
          env,
        }
      : edit
        ? {
            curl: `curl ${base_url}/images/edits \\
  -H "Authorization: Bearer ${key}" \\
  -F model=${model} -F image=@input.png \\
  -F prompt="make it night time" -F response_format=url`,
            python: `import base64
from openai import OpenAI

client = OpenAI(base_url="${base_url}", api_key="${key}")
resp = client.images.edit(
    model="${model}",
    image=open("input.png", "rb"),
    prompt="make it night time",
)
open("out.png", "wb").write(base64.b64decode(resp.data[0].b64_json))`,
            job: `# Async (no ~100s proxy timeout): submit, then poll until "done"
curl ${root}/api/images/jobs \\
  -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \\
  -d "{\\"model\\": \\"${model}\\", \\"prompt\\": \\"make it night time\\", \\"image_b64\\": \\"$(base64 -w0 input.png)\\"}"
curl ${root}/api/images/jobs/JOB_ID -H "Authorization: Bearer ${key}"`,
            env,
          }
        : {
            curl: `curl ${base_url}/images/generations \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "prompt": "a red fox in the snow", "size": "1024x1024", "response_format": "url"}'`,
            python: `import base64
from openai import OpenAI

client = OpenAI(base_url="${base_url}", api_key="${key}")
resp = client.images.generate(
    model="${model}",
    prompt="a red fox in the snow",
    size="1024x1024",
    extra_body={"seed": 42},  # also: steps, cfg, negative_prompt, loras, params
)
open("fox.png", "wb").write(base64.b64decode(resp.data[0].b64_json))`,
            job: `# Async (no ~100s proxy timeout): submit, then poll until "done"
curl ${root}/api/images/jobs \\
  -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "prompt": "a red fox in the snow", "size": "1024x1024"}'
curl ${root}/api/images/jobs/JOB_ID -H "Authorization: Bearer ${key}"`,
            env,
          }
  const tabs = (['curl', 'python', 'job', 'env'] as Tab[]).filter((t) => snippets[t])
  const text = snippets[tab] ?? ''
  const shown = reveal || !key ? text : text.split(key).join('•'.repeat(12))

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
          {tabs.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={`tab ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              type="button"
            >
              {t === 'env' ? '.env' : t === 'job' ? 'async job' : t}
            </button>
          ))}
          <span className="spacer" />
          <CopyButton text={text} />
        </div>
        <pre>{shown}</pre>
      </div>
    </div>
  )
}
