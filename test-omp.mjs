import assert from "node:assert/strict"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { after, test } from "node:test"
import antigravityBridge, { cachePath, toModelConfig, toUsageLimit } from "./omp.mjs"

const FELL_BACK = Symbol.for("magpie.fellBack")
const tempDirs = []

function makeTempDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-omp-test-"))
  tempDirs.push(dir)
  return dir
}

after(() => {
  for (const dir of tempDirs) fs.rmSync(dir, { recursive: true, force: true })
})

test("toModelConfig maps variants, capabilities and limits faithfully", () => {
  const reasoning = toModelConfig({
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    reasoning: true,
    variants: { minimal: {}, low: {}, high: {} },
    capabilities: { attachment: true },
    limit: { context: 1_000_000, output: 65_536 },
  })
  assert.deepEqual(reasoning, {
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    reasoning: true,
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    compat: { forceAdaptiveThinking: true },
    thinkingLevelMap: {
      off: null,
      minimal: "minimal",
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    },
  })

  const raw = toModelConfig({
    id: "raw-model",
    reasoning: true,
    variants: {},
    limit: { context: 0, output: -1 },
  })
  assert.equal(raw.name, "raw-model")
  assert.deepEqual(raw.input, ["text"])
  assert.equal(raw.reasoning, false)
  assert.equal(raw.contextWindow, 256_000)
  assert.equal(raw.maxTokens, 32_768)
  assert.equal(raw.compat, undefined)
  assert.equal(raw.thinkingLevelMap, undefined)
})

test("cachePath honors PI_CODING_AGENT_DIR and falls back to ~/.omp/agent", () => {
  assert.equal(
    cachePath({ PI_CODING_AGENT_DIR: " /custom/agent " }, "/home/tester"),
    path.join("/custom/agent", "antigravity-bridge", "models.json"),
  )
  assert.equal(
    cachePath({ PI_CODING_AGENT_DIR: "   " }, "/home/tester"),
    path.join("/home/tester", ".omp", "agent", "antigravity-bridge", "models.json"),
  )
})

function makeFakeDeps(catalogSequence, fetchImpl = async () => new Response("ok"), usageResult) {
  let index = 0
  let calls = 0
  const plugin = {
    auth: {
      loader: async () => ({ fetch: fetchImpl }),
      usage: async () => usageResult ?? { windows: [], signIn: "kept" },
    },
    provider: {
      models: async () => {
        calls++
        const next = catalogSequence[Math.min(index, catalogSequence.length - 1)]
        index++
        return typeof next === "function" ? next() : next
      },
    },
  }
  const compatCalls = []
  const compat = {
    anthropicMessagesApi: () => ({
      streamSimple: (model, context, options) => {
        compatCalls.push({ model, context, options })
        return { ok: true }
      },
    }),
  }
  const registrations = []
  const pi = {
    registerProvider(id, cfg) {
      registrations.push({ id, cfg })
    },
  }
  return { plugin, compat, pi, registrations, compatCalls, modelCalls: () => calls }
}

test("antigravityBridge registers provider, skips caching fallback catalogs, and refreshes cleanly", async () => {
  const dir = makeTempDir()
  const cacheFile = path.join(dir, "antigravity-bridge", "models.json")

  const fallbackCatalog = {
    "gemini-4-argon": { id: "gemini-4-argon", name: "Fallback", reasoning: true, variants: { high: {} } },
    [FELL_BACK]: true,
  }
  const realCatalog1 = {
    "gemini-3.8-flash": { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", reasoning: true, variants: { low: {}, high: {} } },
  }
  const realCatalog2 = {
    "gemini-4-carbon": { id: "gemini-4-carbon", name: "Gemini 4 Carbon", reasoning: true, variants: { minimal: {}, high: {} } },
  }

  // Fallback on cold start is exposed in-memory but never persisted to disk.
  const coldFallback = makeFakeDeps([fallbackCatalog])
  await antigravityBridge(coldFallback.pi, { plugin: coldFallback.plugin, compat: coldFallback.compat, cacheFile })
  assert.equal(coldFallback.registrations.length, 1)
  assert.equal(coldFallback.registrations[0].id, "antigravity-bridge")
  assert.equal(coldFallback.registrations[0].cfg.api, "antigravity-bridge")
  assert.equal(coldFallback.registrations[0].cfg.baseUrl, "agy://local")
  assert.equal(coldFallback.registrations[0].cfg.apiKey, "agy")
  assert.equal(coldFallback.registrations[0].cfg.models.map(m => m.id).join(","), "gemini-4-argon")
  assert.equal(fs.existsSync(cacheFile), false)

  // Real catalog persists to cache and fetchDynamicModels tracks updates; a
  // fallback or failing discovery must throw so OMP keeps the live list.
  const deps = makeFakeDeps([realCatalog1, realCatalog2, fallbackCatalog, () => { throw new Error("boom") }])
  await antigravityBridge(deps.pi, { plugin: deps.plugin, compat: deps.compat, cacheFile })
  const cfg = deps.registrations.at(-1).cfg
  assert.equal(fs.existsSync(cacheFile), true)
  assert.deepEqual(cfg.models.map(m => m.id), ["gemini-3.8-flash"])

  const refreshed = await cfg.fetchDynamicModels()
  assert.deepEqual(refreshed.map(m => m.id), ["gemini-4-carbon"])
  await assert.rejects(() => cfg.fetchDynamicModels(), /unavailable/)
  await assert.rejects(() => cfg.fetchDynamicModels(), /boom/)
})

test("antigravityBridge uses fresh cache without spawning discovery and refreshes stale cache in background", async () => {
  const dir = makeTempDir()
  const cacheFile = path.join(dir, "models.json")
  const cachedModels = [toModelConfig({ id: "cached-model", reasoning: false })]
  fs.writeFileSync(cacheFile, JSON.stringify({ fetchedAt: Date.now(), models: cachedModels }))

  const freshDeps = makeFakeDeps([{ "fresh-model": { id: "fresh-model" } }])
  await antigravityBridge(freshDeps.pi, { plugin: freshDeps.plugin, compat: freshDeps.compat, cacheFile })
  assert.equal(freshDeps.modelCalls(), 0)
  assert.deepEqual(freshDeps.registrations[0].cfg.models.map(m => m.id), ["cached-model"])

  // Stale cache registers immediately and re-registers once background refresh resolves.
  fs.writeFileSync(cacheFile, JSON.stringify({ fetchedAt: Date.now() - 7 * 60 * 60 * 1000, models: cachedModels }))
  const staleDeps = makeFakeDeps([{ "fresh-model": { id: "fresh-model", name: "Fresh" } }])
  await antigravityBridge(staleDeps.pi, { plugin: staleDeps.plugin, compat: staleDeps.compat, cacheFile })
  assert.deepEqual(staleDeps.registrations[0].cfg.models.map(m => m.id), ["cached-model"])
  for (let i = 0; i < 50 && staleDeps.registrations.length < 2; i++) {
    await new Promise(resolve => setTimeout(resolve, 10))
  }
  assert.equal(staleDeps.registrations.length, 2)
  assert.deepEqual(staleDeps.registrations[1].cfg.models.map(m => m.id), ["fresh-model"])
})

test("streamSimple restores api, compat, thinkingLevelMap and forwards bridge headers/metadata", async () => {
  const dir = makeTempDir()
  const cacheFile = path.join(dir, "models.json")
  const bridgeFetch = async () => new Response("stream")
  const deps = makeFakeDeps([{
    "gemini-3.8-flash": {
      id: "gemini-3.8-flash",
      reasoning: true,
      variants: { minimal: {}, high: {} },
    },
  }], bridgeFetch)

  await antigravityBridge(deps.pi, { plugin: deps.plugin, compat: deps.compat, cacheFile })
  const cfg = deps.registrations[0].cfg
  // Simulate OMP stripping compat, thinkingLevelMap and rewriting api on the model object.
  cfg.streamSimple(
    { id: "gemini-3.8-flash", api: "antigravity-bridge" },
    { messages: [{ role: "user", content: "hi" }] },
    { sessionId: "sess-123", headers: { "x-custom": "1" }, metadata: { extra: true } },
  )
  assert.equal(deps.compatCalls.length, 1)
  const call = deps.compatCalls[0]
  assert.equal(call.model.api, "anthropic-messages")
  assert.deepEqual(call.model.compat, { forceAdaptiveThinking: true })
  assert.equal(call.model.thinkingLevelMap.high, "high")
  assert.equal(call.model.thinkingLevelMap.medium, null)
  assert.equal(call.options.fetch, bridgeFetch)
  assert.deepEqual(call.options.metadata, { extra: true, user_id: "sess-123" })
  assert.equal(call.options.headers["x-custom"], "1")
  assert.equal(call.options.headers["x-magpie-work-dir"], encodeURIComponent(process.cwd()))
})

test("toUsageLimit maps bridge percent windows (0..100) onto OMP percent limits", () => {
  assert.deepEqual(toUsageLimit({ name: "Antigravity Pro · 5 hours", used: 50, span: 5 * 60 * 60, resetsAt: "2026-11-17T12:00:00.000Z" }), {
    id: "antigravity-bridge:antigravity-pro-5-hours",
    label: "Antigravity Pro · 5 hours",
    scope: { provider: "antigravity-bridge", windowId: "5h" },
    window: { id: "5h", label: "5 Hour", durationMs: 18_000_000, resetsAt: Date.parse("2026-11-17T12:00:00.000Z") },
    amount: { unit: "percent", remainingFraction: 0.5, usedFraction: 0.5, remaining: 50, used: 50, limit: 100 },
    status: "ok",
  })
  // Sub-1% values must stay percent-unit (0.5 => 0.5% used, not 50%).
  const subPercent = toUsageLimit({ name: "Antigravity Pro · 5 hours", used: 0.5, span: 5 * 60 * 60 })
  assert.equal(subPercent.amount.used, 0.5)
  assert.equal(subPercent.amount.usedFraction, 0.005)
  assert.equal(subPercent.amount.remaining, 99.5)
  assert.equal(subPercent.amount.remainingFraction, 0.995)
  assert.equal(subPercent.status, "ok")
  // 93.75% used => 6.25% remaining must land in the warning band.
  const warning = toUsageLimit({ name: "Antigravity Pro · 7 days", used: 93.75, span: 7 * 24 * 60 * 60 })
  assert.equal(warning.scope.windowId, "weekly")
  assert.equal(warning.status, "warning")
  assert.equal(warning.window.durationMs, 604_800_000)
  assert.equal(toUsageLimit({ name: "Maxed · 5 hours", used: 100, span: 5 * 60 * 60 }).status, "exhausted")
  for (const bad of [undefined, null, "", -1, 101]) {
    const unknown = toUsageLimit({ name: "Unknown · session", used: bad, display: "Limits not available" })
    assert.equal(unknown.status, "unknown")
    assert.deepEqual(unknown.amount, { unit: "percent" })
    assert.equal(unknown.window.id, "unknown-session")
    assert.equal(unknown.window.resetsAt, undefined)
  }
})

test("registered usage provider reports bridge windows and surfaces reader errors", async () => {
  const dir = makeTempDir()
  const cacheFile = path.join(dir, "models.json")
  const windows = [
    { name: "Antigravity Pro · 5 hours", aside: true, used: 25, span: 5 * 60 * 60, resetsAt: "2026-11-17T12:00:00.000Z" },
    { name: "Unknown · session", aside: true, display: "Limits not available" },
  ]
  const deps = makeFakeDeps([{ "m": { id: "m" } }], undefined, { windows, signIn: "kept" })
  await antigravityBridge(deps.pi, { plugin: deps.plugin, compat: deps.compat, cacheFile })
  const usage = deps.registrations[0].cfg.usage
  assert.equal(usage.id, "antigravity-bridge")
  assert.equal(usage.cacheVersion, 2)
  const report = await usage.fetchUsage()
  assert.equal(report.provider, "antigravity-bridge")
  assert.equal(typeof report.fetchedAt, "number")
  assert.equal(report.limits.length, 2)
  assert.equal(report.limits[0].status, "ok")
  assert.equal(report.limits[1].status, "unknown")

  const failing = makeFakeDeps([{ "m": { id: "m" } }], undefined, { windows: [], error: "agy usage needs authentication.", signIn: "expired" })
  await antigravityBridge(failing.pi, { plugin: failing.plugin, compat: failing.compat, cacheFile: path.join(dir, "other.json") })
  const failingUsage = failing.registrations[0].cfg.usage
  await assert.rejects(() => failingUsage.fetchUsage(), /agy usage needs authentication/)

  const quiet = makeFakeDeps([{ "m": { id: "m" } }])
  await antigravityBridge(quiet.pi, { plugin: quiet.plugin, compat: quiet.compat, cacheFile: path.join(dir, "quiet.json") })
  assert.equal(await quiet.registrations[0].cfg.usage.fetchUsage(), null)
})

test("antigravityBridge suppresses AGY_BRIDGE_EVENTS by default while preserving explicit opt-in", async () => {
  const prev = process.env.AGY_BRIDGE_EVENTS
  const dir = makeTempDir()
  try {
    delete process.env.AGY_BRIDGE_EVENTS
    const a = makeFakeDeps([{ m: { id: "m" } }])
    await antigravityBridge(a.pi, { plugin: a.plugin, compat: a.compat, cacheFile: path.join(dir, "a.json") })
    assert.equal(process.env.AGY_BRIDGE_EVENTS, "0")

    process.env.AGY_BRIDGE_EVENTS = "1"
    const b = makeFakeDeps([{ m: { id: "m" } }])
    await antigravityBridge(b.pi, { plugin: b.plugin, compat: b.compat, cacheFile: path.join(dir, "b.json") })
    assert.equal(process.env.AGY_BRIDGE_EVENTS, "1")
  } finally {
    if (prev === undefined) delete process.env.AGY_BRIDGE_EVENTS
    else process.env.AGY_BRIDGE_EVENTS = prev
  }
})
