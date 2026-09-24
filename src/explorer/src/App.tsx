import { useEffect, useState } from "react"

import { Shell } from "@/components/Shell"
import { type Report, useData } from "@/lib/data"
import { useRoute } from "@/lib/route"
import { Judge } from "@/views/Judge"
import { Live } from "@/views/Live"
import { Method } from "@/views/Method"
import { Overview } from "@/views/Overview"
import { Setups } from "@/views/Setups"

/** Live search exists only behind the local server; the static site has no /api. */
function useLiveAvailable(): boolean {
  const [available, setAvailable] = useState(false)
  useEffect(() => {
    fetch("/api/live")
      .then((response) => setAvailable(response.ok))
      .catch(() => setAvailable(false))
  }, [])
  return available
}

export function App() {
  const route = useRoute()
  const report = useData<Report>("report.json")
  const live = useLiveAvailable()

  let content
  if (report.state === "loading") {
    content = <p className="text-muted-foreground">Loading the report…</p>
  } else if (report.state === "missing") {
    content = (
      <p className="text-muted-foreground">
        No report found. Run <code>groundwork report</code> and <code>groundwork site</code> first.
      </p>
    )
  } else if (route.view === "setups") {
    content = <Setups report={report.value} setup={route.param} question={route.query.get("q")} />
  } else if (route.view === "judge") {
    content = <Judge report={report.value} />
  } else if (route.view === "method") {
    content = <Method report={report.value} />
  } else if (route.view === "live" && live) {
    content = <Live report={report.value} />
  } else {
    content = <Overview report={report.value} />
  }

  return (
    <Shell view={route.view} commit={report.state === "ready" ? report.value.commit : null} live={live}>
      {content}
    </Shell>
  )
}
