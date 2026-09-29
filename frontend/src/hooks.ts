import { useEffect, useRef, useState } from 'react'

/** Poll `fn` every `ms`; returns latest data + error. `tick` forces an immediate refetch. */
export function usePoll<T>(fn: () => Promise<T>, ms: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [tick, setTick] = useState(0)
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const run = async () => {
      try {
        const d = await fnRef.current()
        if (alive) {
          setData(d)
          setError(null)
        }
      } catch (e) {
        if (alive) setError(e as Error)
      }
      if (alive) timer = setTimeout(run, ms)
    }
    run()
    return () => {
      alive = false
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, tick, ...deps])

  return { data, error, refresh: () => setTick((t) => t + 1) }
}
