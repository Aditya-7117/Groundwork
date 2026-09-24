import { Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"

const KEY = "groundwork-theme"

function stored(): "light" | "dark" | null {
  try {
    const value = window.localStorage.getItem(KEY)
    return value === "light" || value === "dark" ? value : null
  } catch {
    return null
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">(
    () => stored() ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
  )
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    try {
      window.localStorage.setItem(KEY, theme)
    } catch {
      // Storage can be blocked; the theme then lasts for this visit only.
    }
  }, [theme])
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  )
}
