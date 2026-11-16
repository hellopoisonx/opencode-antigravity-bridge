import assert from "node:assert/strict"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { after, test } from "node:test"
import antigravityBridge, { cachePath, toModelConfig } from "./omp.mjs"

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

function makeFakeDeps(catalogSequence, fetchImpl = async () => new Response("ok")) {
  let index = 0
  let calls = 0
  const plugin = {
    auth: { loader: async () => ({ fetch: fetchImpl }) },
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

  // Real catalog persists to cache and refreshModels tracks updates + preserves non-fallback models.
  const deps = makeFakeDeps([realCatalog1, realCatalog2, fallbackCatalog, () => { throw new Error("boom") }])
  await antigravityBridge(deps.pi, { plugin: deps.plugin, compat: deps.compat, cacheFile })
  const cfg = deps.registrations.at(-1).cfg
  assert.equal(fs.existsSync(cacheFile), true)
  assert.deepEqual(cfg.models.map(m => m.id), ["gemini-3.8-flash"])
  assert.deepEqual((await cfg.refreshModels({ allowNetwork: false })).map(m => m.id), ["gemini-3.8-flash"])

  const refreshed = await cfg.refreshModels({ allowNetwork: true })
  assert.deepEqual(refreshed.map(m => m.id), ["gemini-4-carbon"])
  // Non-network restore must reflect the latest refreshed list, not the initial snapshot.
  assert.deepEqual((await cfg.refreshModels({ allowNetwork: false })).map(m => m.id), ["gemini-4-carbon"])
  // Fallback or throwing refresh keeps the current non-fallback list intact.
  assert.deepEqual((await cfg.refreshModels({ allowNetwork: true })).map(m => m.id), ["gemini-4-carbon"])
  assert.deepEqual((await cfg.refreshModels({ allowNetwork: true })).map(m => m.id), ["gemini-4-carbon"])
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
