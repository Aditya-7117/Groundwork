import { Activity, BookOpen, Gauge, Layers, Scale, Search } from "lucide-react"
import type { ReactNode } from "react"

import { ThemeToggle } from "@/components/ThemeToggle"
import { href } from "@/lib/route"
import { cn } from "@/lib/utils"

const NAV = [
  { view: "overview", label: "Overview", icon: Gauge },
  { view: "setups", label: "Setups", icon: Layers },
  { view: "judge", label: "Judge study", icon: Scale },
  { view: "method", label: "Method", icon: BookOpen },
  { view: "live", label: "Live search", icon: Search },
]

export function Shell({
  view,
  commit,
  live,
  children,
}: {
  view: string
  commit: string | null
  live: boolean
  children: ReactNode
}) {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="hidden w-56 shrink-0 border-r bg-muted/30 md:flex md:flex-col">
        <a href={href("overview")} className="flex items-center gap-2 px-5 py-5">
          <Activity className="size-5 text-primary" />
          <span className="font-semibold tracking-tight">Groundwork</span>
        </a>
        <nav className="flex flex-col gap-1 px-3">
          {NAV.filter((item) => item.view !== "live" || live).map((item) => (
            <a
              key={item.view}
              href={href(item.view)}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-muted hover:text-foreground",
                view === item.view && "bg-muted font-medium text-foreground",
              )}
            >
              <item.icon className="size-4" />
              {item.label}
            </a>
          ))}
        </nav>
        <p className="mt-auto px-5 py-4 text-xs text-muted-foreground">
          Retrieval evaluation on Natural Questions
          {commit && <span className="mt-1 block font-mono">commit {commit.slice(0, 7)}</span>}
        </p>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b px-4 py-3 md:px-8">
          <nav className="flex gap-3 text-sm md:hidden">
            {NAV.filter((item) => item.view !== "live" || live).map((item) => (
              <a key={item.view} href={href(item.view)} className={cn(view === item.view && "font-medium")}>
                {item.label}
              </a>
            ))}
          </nav>
          <span className="hidden text-sm text-muted-foreground md:block">
            30 retrieval setups · 3,220 questions · passage-level relevance
          </span>
          <ThemeToggle />
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 md:px-8">{children}</main>
      </div>
    </div>
  )
}
