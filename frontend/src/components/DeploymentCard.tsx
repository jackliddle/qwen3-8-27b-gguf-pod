import { useState } from 'react'
import { api, fmtBytes, post, type Deployment } from '../api'
import { EndpointPanel } from './EndpointPanel'
import { LogViewer } from './LogViewer'

const BUSY = new Set(['pending', 'downloading', 'starting', 'stopping'])

function elapsed(from: number, to?: number | null) {
  const s = Math.max(0, Math.round((to ?? Date.now() / 1000) - from))
  return s < 90 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
}

export function DeploymentCard({ dep, onChange, onError }: { dep: Deployment; onChange: () => void; onError: (e: unknown) => void }) {
  const [showLogs, setShowLogs] = useState(dep.status !== 'ready' && dep.status !== 'stopped')
  const [busy, setBusy] = useState(false)
  const active = !['stopped', 'failed'].includes(dep.status)

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
    } catch (e) {
      onError(e)
    } finally {
      setBusy(false)
      onChange()
    }
  }

  const pct = dep.progress?.total_bytes ? (dep.progress.done_bytes / dep.progress.total_bytes) * 100 : null
  const params = Object.entries(dep.params)

  return (
    <article className={`card dep status-${dep.status}`}>
      <header className="dep-head">
        <div>
          <h3>{dep.name}</h3>
          <div className="meta">
            <span className="chip">{dep.engine}</span>
            <code>{dep.served_name}</code>
            <span className="muted">
              {dep.status === 'ready' && dep.ready_at ? `up ${elapsed(dep.ready_at)}` : BUSY.has(dep.status) ? elapsed(dep.created_at) : ''}
            </span>
          </div>
        </div>
        <span className={`badge badge-${dep.status}`}>
          {BUSY.has(dep.status) && <span className="spinner" aria-hidden />}
          {dep.status}
        </span>
      </header>

      {dep.status === 'downloading' && pct != null && (
        <div className="progress" aria-label="download progress">
          <div className="bar" style={{ width: `${pct}%` }} />
          <span>
            {fmtBytes(dep.progress?.done_bytes)} / {fmtBytes(dep.progress?.total_bytes)} · {pct.toFixed(1)}%
          </span>
        </div>
      )}

      {dep.error && <p className="error-text">{dep.error}</p>}

      {params.length > 0 && (
        <div className="params">
          {params.map(([k, v]) => (
            <span key={k} className="param">
              {k}=<b>{String(v)}</b>
            </span>
          ))}
        </div>
      )}

      {dep.status === 'ready' && <EndpointPanel dep={dep} />}

      <footer className="actions">
        {active ? (
          <>
            <button className="btn btn-danger" disabled={busy || dep.status === 'stopping'} onClick={() => act(() => post(`/api/deployments/${dep.id}/stop`))}>
              Stop &amp; delete files
            </button>
            <button
              className="btn"
              disabled={busy || dep.status === 'stopping'}
              title="Stop but keep the downloaded files, so a restart doesn't re-download"
              onClick={() => act(() => post(`/api/deployments/${dep.id}/stop?delete_files=false`))}
            >
              Stop, keep files
            </button>
            {dep.status === 'ready' && (
              <button className="btn" disabled={busy} onClick={() => act(() => post(`/api/deployments/${dep.id}/restart`))}>
                Restart
              </button>
            )}
          </>
        ) : (
          <>
            <button className="btn btn-primary" disabled={busy} onClick={() => act(() => post(`/api/deployments/${dep.id}/restart`))}>
              Redeploy
            </button>
            {dep.status === 'failed' && (
              <button className="btn btn-danger" disabled={busy} title="Delete any partially downloaded files" onClick={() => act(() => post(`/api/deployments/${dep.id}/stop`))}>
                Delete files
              </button>
            )}
            <button className="btn btn-ghost" disabled={busy} onClick={() => act(() => api(`/api/deployments/${dep.id}`, { method: 'DELETE' }))}>
              Dismiss
            </button>
          </>
        )}
        <span className="spacer" />
        <button className="btn btn-ghost" onClick={() => setShowLogs(!showLogs)} aria-expanded={showLogs}>
          {showLogs ? 'Hide logs' : 'Logs'}
        </button>
      </footer>

      {showLogs && <LogViewer depId={dep.id} />}
    </article>
  )
}
