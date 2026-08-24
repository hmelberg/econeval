// scopedStore(store, modelName) -> a store-shaped wrapper whose .get().model is
// store.get().model.models[modelName], and whose .applyOp maps fn over that same sub-model,
// splicing the result back into a fresh top-level model before handing it to the real store's
// applyOp (which does the actual serialize/reparse/commit/undo-snapshot work). Everything else
// (select/undo/redo/markSaved/subscribe) passes straight through — undo history, dirty state and
// selection are document-wide, not per-sub-model (constraints.md: "Undo/redo = document text
// snapshots"). Composable: scopedStore(scopedStore(store, 'a'), 'b') correctly reaches
// model.models.a.models.b, so entering nested sub-models is just chaining this wrapper once per
// currentModelPath segment (see scopedStoreFor below) — no separate multi-level-aware variant
// needed.

export function scopedStore(store, modelName) {
  return {
    get() {
      const outer = store.get();
      const sub = outer.model && outer.model.models ? outer.model.models[modelName] : undefined;
      return { ...outer, model: sub ?? null };
    },
    applyOp(fn, opts) {
      store.applyOp((model) => {
        const sub = model.models && model.models[modelName];
        if (!sub) throw new Error(`scopedStore: sub-model '${modelName}' not found`);
        const newSub = fn(sub);
        return { ...model, models: { ...model.models, [modelName]: newSub } };
      }, opts);
    },
    // Task 10 controller ruling: stamp this wrapper's own name onto selection.modelPath so
    // store.js's isSelectionValid can resolve the SCOPED model the selection actually refers to
    // (rather than checking, say, a sub-model's state name against the top-level model). Chained
    // wrappers each prepend their own name, so scopedStore(scopedStore(s,'outer'),'inner') ends
    // up stamping ['outer','inner'] on the base store's selection — matching how canvas.js builds
    // currentModelPath (outer pushed before inner).
    select: (sel) => store.select({ ...sel, modelPath: [modelName, ...(sel.modelPath ?? [])] }),
    undo: () => store.undo(),
    redo: () => store.redo(),
    markSaved: () => store.markSaved(),
    subscribe: (listener) => store.subscribe(listener),
  };
}

export function scopedStoreFor(baseStore, path) {
  return path.reduce((s, name) => scopedStore(s, name), baseStore);
}
