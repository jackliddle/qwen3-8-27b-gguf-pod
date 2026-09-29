import { useEffect, useState } from 'react'
import { api, ApiError, fmtBytes, getKey, post, setKey, type Deployment, type Recipe, type Status } from './api'
import { CopyButton } from './components/CopyButton'
import { DeployDialog } from './components/DeployDialog'
import { DeploymentCard } from './components/DeploymentCard'
import { PushRecipeDialog } from './components/PushRecipeDialog'
import { usePoll } from './hooks'

function Login({ onLogin }: { onLogin: (k: string) => void }) {
  const [key, setK] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await api('/api/status', {}, key)
      onLogin(key)
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? 'Wrong API key' : String(err))
    }
  }
  return (
    <main className="login">
      <form className="card" onSubmit={submit}>
        <h1 className="brand">
          <span className="logo">M</span> Model Pod
        </h1>
        <label className="field">
          <span className="field-name">API key</span>
          <input type="password" autoFocus value={key} onChange={(e) => setK(e.target.value)} placeholder="the pod's API_KEY" />
        </label>
        {error && <p className="error-text">{error}</p>}
        <button className="btn btn-primary" type="submit" disabled={!key}>
          Sign in
        </button>
      </form>
    </main>
  )
}

function StatusBar({ status }: { status: Status | null }) {
  if (!status) return null
  const disk = status.disk
  return (
    <div className="statusbar">
      {status.gpus.length === 0 && <span className="stat muted">No GPU detected</span>}
      {status.gpus.map((g) => {
        const pct = (g.memory_used_mib / g.memory_total_mib) * 100
        return (
          <span className="stat" key={g.index} title={`GPU ${g.index}`}>
            <b>{g.name.replace('NVIDIA ', '')}</b>
            <span className="meter">
              <span style={{ width: `${pct}%` }} className={pct > 90 ? 'hot' : ''} />
            </span>
            {(g.memory_used_mib / 1024).toFixed(1)}/{(g.memory_total_mib / 1024).toFixed(0)} GB
          </span>
        )
      })}
      <span className="stat">
        <b>Disk</b> {fmtBytes(disk.free_bytes)} free
      </span>
      <span className="stat openai">
        <b>OpenAI base</b> <code>{status.openai_base_url}</code>
        <CopyButton text={status.openai_base_url} />
      </span>
    </div>
  )
}

function Dashboard({ onLogout }: { onLogout: () => void }) {
  const status = usePoll(() => api<Status>('/api/status'), 5000)
  const deps = usePoll(() => api<Deployment[]>('/api/deployments'), 2000)
  const recipes = usePoll(() => api<Recipe[]>('/api/recipes'), 10000)
  const [deploying, setDeploying] = useState<Recipe | null>(null)
  const [pushing, setPushing] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  const onError = (e: unknown) => {
    if (e instanceof ApiError && e.status === 401) return onLogout()
    setToast(e instanceof Error ? e.message : String(e))
    setTimeout(() => setToast(null), 6000)
  }
  const refreshAll = () => {
    deps.refresh()
    recipes.refresh()
  }
  const unauthorized = status.error instanceof ApiError && status.error.status === 401
  useEffect(() => {
    if (unauthorized) onLogout()
  }, [unauthorized, onLogout])

  const deployments = [...(deps.data ?? [])].sort((a, b) => b.created_at - a.created_at)
  const shownRecipes = (recipes.data ?? []).filter((r) =>
    `${r.name} ${r.id} ${r.engine} ${r.capabilities.join(' ')}`.toLowerCase().includes(filter.toLowerCase()),
  )

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand">
          <span className="logo">M</span> Model Pod
          {status.data?.pod_id && <span className="muted pod-id">{status.data.pod_id}</span>}
          {status.data?.dev && <span className="chip">dev</span>}
        </h1>
        <span className="spacer" />
        <button className="btn btn-ghost" onClick={onLogout}>
          Sign out
        </button>
      </header>
      <StatusBar status={status.data} />
      {status.error && !(status.error instanceof ApiError) && <div className="banner">Supervisor unreachable: {status.error.message}</div>}

      <main className="content">
        <section>
          <div className="section-head">
            <h2>Deployments</h2>
          </div>
          {deployments.length === 0 ? (
            <p className="empty">Nothing deployed. Pick a model below.</p>
          ) : (
            <div className="dep-list">
              {deployments.map((d) => (
                <DeploymentCard key={`${d.id}-${d.created_at}`} dep={d} onChange={refreshAll} onError={onError} />
              ))}
            </div>
          )}
        </section>

        <section>
          <div className="section-head">
            <h2>Models</h2>
            <input className="search" placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
            <span className="spacer" />
            <button
              className="btn btn-ghost"
              title={status.data?.recipes_repo ? `Re-fetch recipes from ${status.data.recipes_repo}` : 'Reload recipes from disk'}
              onClick={async () => {
                try {
                  const r = await post<{ remote_error: string | null }>('/api/recipes/refresh')
                  if (r.remote_error) onError(new Error(`Remote recipes: ${r.remote_error}`))
                } catch (e) {
                  onError(e)
                }
                recipes.refresh()
              }}
            >
              Refresh
            </button>
            <button className="btn" onClick={() => setPushing(true)}>
              Add recipe
            </button>
          </div>
          {status.data?.recipes_remote_error && <p className="error-text">Remote recipe fetch failed: {status.data.recipes_remote_error}</p>}
          <div className="recipe-grid">
            {shownRecipes.map((r) => (
              <article className="card recipe" key={r.id}>
                <header>
                  <h3>{r.name}</h3>
                  <span className={`chip engine-${r.engine}`}>{r.engine}</span>
                </header>
                {r.description && <p className="muted desc">{r.description}</p>}
                <div className="facts">
                  <span>
                    <b>{r.vram_gb}</b> GB VRAM
                  </span>
                  {r.disk_gb != null && (
                    <span>
                      <b>{r.disk_gb}</b> GB disk
                    </span>
                  )}
                  <span className="caps">
                    {r.capabilities.map((c) => (
                      <span key={c} className="cap">
                        {c}
                      </span>
                    ))}
                  </span>
                </div>
                <footer className="actions">
                  <code className="muted small">{r.id}</code>
                  {r.origin !== 'builtin' && <span className="chip subtle">{r.origin}</span>}
                  <span className="spacer" />
                  {r.deployment_status && !['stopped', 'failed'].includes(r.deployment_status) ? (
                    <span className={`badge badge-${r.deployment_status}`}>{r.deployment_status}</span>
                  ) : (
                    <button className="btn btn-primary" onClick={() => setDeploying(r)}>
                      Deploy
                    </button>
                  )}
                </footer>
              </article>
            ))}
          </div>
        </section>
      </main>

      {deploying && <DeployDialog recipe={deploying} onClose={() => setDeploying(null)} onDeployed={refreshAll} />}
      {pushing && <PushRecipeDialog onClose={() => setPushing(false)} onPushed={recipes.refresh} />}
      {toast && (
        <div className="toast" role="alert" onClick={() => setToast(null)}>
          {toast}
        </div>
      )}
    </div>
  )
}

/** One-click login links (`/#key=...`, printed by scripts/pod.sh). The fragment
 * never reaches the server; it's moved to localStorage and cleared from the URL. */
function keyFromHash(): string {
  const m = window.location.hash.match(/key=([^&]+)/)
  if (!m) return getKey()
  const key = decodeURIComponent(m[1])
  setKey(key)
  history.replaceState(null, '', window.location.pathname + window.location.search)
  return key
}

export default function App() {
  const [key, setKeyState] = useState(keyFromHash)
  if (!key)
    return (
      <Login
        onLogin={(k) => {
          setKey(k)
          setKeyState(k)
        }}
      />
    )
  return (
    <Dashboard
      onLogout={() => {
        setKey(null)
        setKeyState('')
      }}
    />
  )
}
