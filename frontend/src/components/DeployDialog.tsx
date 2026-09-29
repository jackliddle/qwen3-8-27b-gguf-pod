import { useState } from 'react'
import { ApiError, post, type Recipe } from '../api'

export function DeployDialog({ recipe, onClose, onDeployed }: { recipe: Recipe; onClose: () => void; onDeployed: () => void }) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(recipe.params).map(([k, s]) => [k, s.default == null ? '' : String(s.default)])),
  )
  const [error, setError] = useState<string | null>(null)
  const [needsForce, setNeedsForce] = useState(false)
  const [busy, setBusy] = useState(false)

  const submit = async (force: boolean) => {
    setBusy(true)
    setError(null)
    // Only send changed values; the server fills in defaults.
    const params = Object.fromEntries(
      Object.entries(values).filter(([k, v]) => v !== (recipe.params[k].default == null ? '' : String(recipe.params[k].default))),
    )
    try {
      await post('/api/deployments', { recipe_id: recipe.id, params, force })
      onDeployed()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setNeedsForce(e instanceof ApiError && e.status === 409 && e.message.includes('VRAM'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay" role="dialog" aria-modal aria-labelledby="deploy-title" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form
        className="dialog"
        onSubmit={(e) => {
          e.preventDefault()
          submit(false)
        }}
      >
        <h2 id="deploy-title">Deploy {recipe.name}</h2>
        <p className="muted">
          {recipe.engine} · serves as <code>{recipe.served_name}</code> · ~{recipe.vram_gb} GB VRAM
          {recipe.disk_gb ? ` · ~${recipe.disk_gb} GB download` : ''}
        </p>

        {Object.keys(recipe.params).length === 0 && <p className="muted">This recipe has no configurable options.</p>}

        <div className="form-grid">
          {Object.entries(recipe.params).map(([name, spec]) => (
            <label key={name} className="field">
              <span className="field-name">
                {name}
                {spec.flag && <code className="flag">{spec.flag}</code>}
              </span>
              {spec.type === 'enum' ? (
                <select value={values[name]} onChange={(e) => setValues({ ...values, [name]: e.target.value })}>
                  {spec.values?.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              ) : spec.type === 'bool' ? (
                <select value={values[name]} onChange={(e) => setValues({ ...values, [name]: e.target.value })}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : (
                <input
                  type={spec.type === 'str' ? 'text' : 'number'}
                  step={spec.type === 'float' ? 'any' : '1'}
                  value={values[name]}
                  onChange={(e) => setValues({ ...values, [name]: e.target.value })}
                />
              )}
              {spec.description && <small className="muted">{spec.description}</small>}
            </label>
          ))}
        </div>

        {recipe.extra_args.length > 0 && (
          <details className="extra">
            <summary>Fixed engine args</summary>
            <code>{recipe.extra_args.join(' ')}</code>
          </details>
        )}

        {error && <p className="error-text">{error}</p>}

        <div className="actions">
          <span className="spacer" />
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Cancel
          </button>
          {needsForce && (
            <button type="button" className="btn btn-danger" disabled={busy} onClick={() => submit(true)}>
              Deploy anyway
            </button>
          )}
          <button type="submit" className="btn btn-primary" disabled={busy}>
            {busy ? 'Deploying…' : 'Deploy'}
          </button>
        </div>
      </form>
    </div>
  )
}
