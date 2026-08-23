// Model registry: versioned local storage + autosave. Pure over an injected storage object
// ({getItem, setItem, removeItem} — the same shape as window.localStorage), per constraints.md
// ("pure logic ... lives in DOM-free modules with tests"). app.js is the only DOM-facing caller;
// it injects globalThis.localStorage.
//
// Storage layout (two independent keys, so autosave never touches the version history and a
// corrupt registry never touches the autosave draft):
//   'econeval.models.v1'   -> JSON { [id]: { name, versions: [{ts, label, text}, ...] } }
//                             versions are newest-first (prepended on every saveVersion), capped
//                             at 20 per model (oldest evicted past the cap).
//   'econeval.autosave.v1' -> raw text (the current document text, not JSON-wrapped — there's
//                             nothing to structure beyond the text itself).

const MODELS_KEY = 'econeval.models.v1';
const AUTOSAVE_KEY = 'econeval.autosave.v1';
const VERSION_CAP = 20;

// No uncontrolled randomness anywhere under js/ (test/golden.test.js enforces this repo-wide —
// PSA's own seeded RNG is the only randomness source the app is allowed). crypto.randomUUID() is
// available in every evergreen browser and Node 19+ (including over plain http://localhost, which
// browsers treat as a secure context); the fallback below is a monotonic counter, not a random
// one, for the vanishingly unlikely environment without crypto.randomUUID at all.
let idCounter = 0;
function defaultGenId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  idCounter += 1;
  return `m${Date.now().toString(36)}-${idCounter}`;
}

// Bad/missing/non-object JSON -> {} (an empty registry), never a thrown error — a corrupt or
// pre-schema blob in storage must not crash the app; the New/Open dialogs just see nothing saved
// yet. A well-formed-but-implausible value (an array, a primitive) is treated the same way.
function readIndex(storage) {
  let raw;
  try {
    raw = storage.getItem(MODELS_KEY);
  } catch {
    return {};
  }
  if (raw == null) return {};
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
  return parsed;
}

function writeIndex(storage, index) {
  storage.setItem(MODELS_KEY, JSON.stringify(index));
}

export function createRegistry(storage, { now = Date.now, genId = defaultGenId } = {}) {
  return {
    list() {
      const index = readIndex(storage);
      return Object.entries(index)
        .map(([id, entry]) => ({
          id,
          name: entry.name,
          updated: entry.versions[0]?.ts ?? 0,
          versionCount: entry.versions.length,
        }))
        .sort((a, b) => b.updated - a.updated); // most recently saved model first
    },

    saveVersion(id, name, text, label) {
      const index = readIndex(storage);
      const useId = id ?? genId();
      // A given id that isn't (or no longer is) in the index gets created fresh rather than
      // throwing — lenient upsert. A stale id surviving a registry reset/corruption recovery
      // shouldn't turn a routine "Save version" into a crash; it just starts that model's history
      // over under the id the app already had in hand.
      const existing = index[useId];
      const versions = existing ? existing.versions : [];
      const entry = {
        name,
        versions: [{ ts: now(), label: label ?? null, text }, ...versions].slice(0, VERSION_CAP),
      };
      index[useId] = entry;
      writeIndex(storage, index);
      return useId;
    },

    // Not in the brief's literal method list, but needed to build the "Open" dialog's version
    // picker (list registry + versions + delete): list() alone exposes only a version COUNT, and
    // load(id, versionTs) needs a ts handed to it from somewhere. Text is deliberately omitted
    // (the dialog only needs ts/label to render a picker; the full text is fetched via load()
    // only once a specific version is actually chosen).
    listVersions(id) {
      const index = readIndex(storage);
      const entry = index[id];
      if (!entry) return [];
      return entry.versions.map((v) => ({ ts: v.ts, label: v.label ?? null }));
    },

    load(id, versionTs) {
      const index = readIndex(storage);
      const entry = index[id];
      if (!entry) throw new Error(`load: no such model "${id}"`);
      const version = versionTs == null
        ? entry.versions[0]
        : entry.versions.find((v) => v.ts === versionTs);
      if (!version) throw new Error(`load: no such version for model "${id}"`);
      return { text: version.text, name: entry.name };
    },

    remove(id) {
      const index = readIndex(storage);
      if (!(id in index)) return; // already gone: a safe no-op, not an error
      delete index[id];
      writeIndex(storage, index);
    },

    autosave(text) {
      try {
        storage.setItem(AUTOSAVE_KEY, text);
      } catch {
        // Best-effort only (e.g. a full/blocked storage quota) — autosave failing silently is
        // strictly better than crashing the app on every keystroke; the user's explicit "Save
        // version" path is the durable one.
      }
    },

    readAutosave() {
      try {
        const raw = storage.getItem(AUTOSAVE_KEY);
        return raw == null ? null : raw;
      } catch {
        return null;
      }
    },
  };
}
