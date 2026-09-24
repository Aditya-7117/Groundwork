// Hash routing: "#/setups/fixed-bm25?q=123". It works on a static host with no server rewrites.

import { useEffect, useState } from "react"

export interface Route {
  view: string
  param: string | null
  query: URLSearchParams
}

function parse(hash: string): Route {
  const [path, search = ""] = hash.replace(/^#\/?/, "").split("?")
  const [view = "overview", param = null] = path.split("/").filter(Boolean)
  return { view, param, query: new URLSearchParams(search) }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash))
  useEffect(() => {
    const update = () => setRoute(parse(window.location.hash))
    window.addEventListener("hashchange", update)
    return () => window.removeEventListener("hashchange", update)
  }, [])
  return route
}

export const href = (view: string, param?: string, query?: Record<string, string>): string => {
  const search = query ? `?${new URLSearchParams(query).toString()}` : ""
  return `#/${view}${param ? `/${encodeURIComponent(param)}` : ""}${search}`
}
