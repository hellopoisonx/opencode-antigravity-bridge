// The OMP (oh-my-pi) face of the bridge.
//
// OMP registers a provider as { api, baseUrl, models, fetchDynamicModels,
// streamSimple, usage }; usage feeds `/usage` and `omp usage`.
// The bridge already implements the Anthropic Messages wire protocol in-process
// and serves it from `agy acp`, so this file only translates: OMP's own
// Anthropic adapter builds the request and parses the stream, and the bridge's
// fetch answers it.
//
// Model discovery runs `agy models` (~3 s), so the list is cached next to the
// host's agent dir and refreshed in the background; a cold start awaits it
// once. `omp models refresh` reuses the same path.

import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { AntigravityBridgePlugin } from "./index.mjs"

const PROVIDER_ID = "antigravity-bridge"
// OMP reserves built-in API names and rejects a custom streamSimple under one.
const API = "antigravity-bridge"
// The adapter maps request options and reads model.compat by API, so it must
// see a built-in API and a compat object; OMP may strip both from extension
// models, so streamSimple restores them from this package's own catalog.
const ADAPTER_API = "anthropic-messages"
const BASE_URL = "agy://local"
const DEFAULT_CONTEXT = 256_000
const DEFAULT_OUTPUT = 32_768
const LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
const COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
const CACHE_TTL_MS = 6 * 60 * 60 * 1000
const FELL_BACK = Symbol.for("magpie.fellBack")

let pending

function bridge() {
  return (pending ??= AntigravityBridgePlugin({}, {}))
}

export function cachePath(env = process.env, home = os.homedir()) {
  const dir = env.PI_CODING_AGENT_DIR?.trim() || path.join(home, ".omp", "agent")
  return path.join(dir, "antigravity-bridge", "models.json")
}

// One bridge catalog entry as an OMP model. Only efforts the agy catalog
// really advertises stay selectable; a raw (unvariant) id has no effort knob.
export function toModelConfig(item) {
  const efforts = Object.keys(item.variants ?? {})
  const reasoning = item.reasoning === true && efforts.length > 0
  const context = Number(item.limit?.context)
  const output = Number(item.limit?.output)
  const model = {
    id: item.id,
    name: item.name || item.id,
    input: item.capabilities?.attachment ? ["text", "image"] : ["text"],
    cost: COST,
    reasoning,
    contextWindow: context > 0 ? context : DEFAULT_CONTEXT,
    maxTokens: output > 0 ? output : DEFAULT_OUTPUT,
  }
  if (!reasoning) return model
  model.compat = { forceAdaptiveThinking: true }
  model.thinkingLevelMap = Object.fromEntries(LEVELS.map(level => [level, efforts.includes(level) ? level : null]))
  return model
}

// agy's two quota windows, as OMP window descriptors.
const WINDOW_BY_SPAN = new Map([
  [5 * 60 * 60, { id: "5h", label: "5 Hour", durationMs: 5 * 3_600_000 }],
  [7 * 24 * 60 * 60, { id: "weekly", label: "Weekly", durationMs: 7 * 86_400_000 }],
])

function slug(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "quota"
}

// One bridge quota window as an OMP usage limit. agy reports fractions of an
// unnamed allowance, so amounts are percent-unit; quota it cannot measure
// stays visible as unknown instead of being reported as exhausted.
export function toUsageLimit(window) {
  const used = Number(window.used)
  const known = Number.isFinite(used) && used >= 0 && used <= 1
  const remaining = known ? 1 - used : undefined
  const period = WINDOW_BY_SPAN.get(window.span) ?? { id: slug(window.name), label: window.name }
  const resetsAt = window.resetsAt ? Date.parse(window.resetsAt) : undefined
  return {
    id: `${PROVIDER_ID}:${slug(window.name)}`,
    label: window.name,
    scope: { provider: PROVIDER_ID, windowId: period.id },
    window: { ...period, ...(Number.isFinite(resetsAt) ? { resetsAt } : {}) },
    amount: known
      ? { unit: "percent", remainingFraction: remaining, usedFraction: used, remaining: remaining * 100, used: used * 100, limit: 100 }
      : { unit: "percent" },
    status: !known ? "unknown" : remaining <= 0 ? "exhausted" : remaining <= 0.1 ? "warning" : "ok",
  }
}

export async function listModels(signal, pluginOverride) {
  const plugin = pluginOverride ?? await bridge()
  const catalog = await plugin.provider.models(undefined, {})
  if (signal?.aborted) return []
  const list = Object.values(catalog).map(toModelConfig)
  Object.defineProperty(list, FELL_BACK, { value: Boolean(catalog[FELL_BACK]) })
  return list
}

function readCache(file = cachePath()) {
  try {
    const cached = JSON.parse(fs.readFileSync(file, "utf8"))
    if (!Array.isArray(cached?.models) || cached.models.length === 0) return null
    if (!Number.isFinite(cached?.fetchedAt)) return null
    return cached
  } catch {
    return null
  }
}

function writeCache(models, file = cachePath()) {
  if (!Array.isArray(models) || !models.length || models[FELL_BACK]) return
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, JSON.stringify({ fetchedAt: Date.now(), models }))
  } catch { }
}

export default async function antigravityBridge(pi, deps = {}) {
  const compat = deps.compat ?? await import("@earendil-works/pi-ai/compat")
  const innerStream = typeof compat.anthropicMessagesApi === "function"
    ? compat.anthropicMessagesApi().streamSimple
    : compat.streamSimple
  const plugin = deps.plugin ?? await bridge()
  const { fetch: bridgeFetch } = await plugin.auth.loader(async () => ({ type: "api", key: "agy" }))
  const file = deps.cacheFile ?? cachePath()

  let currentModels = []
  let byId = new Map()
  const applyModels = (list, persist) => {
    currentModels = list
    byId = new Map(list.map(model => [model.id, model]))
    if (persist) writeCache(list, file)
  }

  const cached = readCache(file)
  if (cached) {
    applyModels(cached.models, false)
  } else {
    const initial = await listModels(undefined, plugin).catch(() => [])
    applyModels(initial, true)
  }

  const register = list => pi.registerProvider(PROVIDER_ID, {
    name: "Antigravity Bridge",
    baseUrl: BASE_URL,
    api: API,
    // agy owns auth; this literal only satisfies OMP's provider contract.
    apiKey: "agy",
    models: list,
    // OMP's native discovery seam (`omp models refresh`, 15 s bound, cached a
    // day). A fallback or failed fetch must throw, not return a stale list:
    // the registry caches whatever comes back.
    async fetchDynamicModels(signal) {
      const fresh = await listModels(signal, plugin)
      if (fresh.length === 0 || fresh[FELL_BACK] && currentModels.length) {
        throw new Error("agy model discovery is unavailable")
      }
      applyModels(fresh, true)
      return fresh
    },
    // `/usage` and `omp usage`: OMP fetches this like any provider's quota.
    // A thrown error is cached as a failed refresh with backoff, so it only
    // carries the reader's own message; empty windows mean "no report".
    usage: {
      id: PROVIDER_ID,
      cacheVersion: 1,
      async fetchUsage() {
        const { windows, error } = await plugin.auth.usage()
        if (!windows?.length) {
          if (error) throw new Error(error)
          return null
        }
        return { provider: PROVIDER_ID, fetchedAt: Date.now(), limits: windows.map(toUsageLimit) }
      },
    },
    streamSimple(model, context, options) {
      const mine = byId.get(model.id)
      const thinkingLevelMap = mine?.thinkingLevelMap ?? model.thinkingLevelMap
      return innerStream({
        ...model,
        api: ADAPTER_API,
        compat: mine?.compat ?? model.compat ?? {},
        ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
      }, context, {
        ...options,
        fetch: bridgeFetch,
        // The conversation key includes this: two sessions sharing a first
        // message must not share a parked agy session.
        metadata: { ...options?.metadata, user_id: options?.sessionId ?? "omp" },
        headers: { ...options?.headers, "x-magpie-work-dir": encodeURIComponent(process.cwd()) },
      })
    },
  })
  register(currentModels)

  if (cached && Date.now() - cached.fetchedAt > CACHE_TTL_MS) {
    void listModels(undefined, plugin).then(fresh => {
      if (fresh.length > 0 && !fresh[FELL_BACK]) {
        applyModels(fresh, true)
        register(fresh)
      }
    }).catch(() => { })
  }
}
