// The right-click menu (Task 8) — restores the only pointer-driven route to delete a node/edge
// that the 4-tool toolbar's deletion took with it (Task 5 deleted the toolbar; Delete/Backspace on
// a selection still works, but there was no mouse route left). A single, DOM-only component: it
// builds the menu, positions it, and tears it down; it holds no business logic of its own
// (constraints.md: "DOM modules hold no business logic") — every item's `action` is a closure the
// CALLER (canvas/index.js) built, already carrying whatever store/model/flush/runOp it needs. This
// module never imports ops.js or touches the store.
//
// createContextMenu() -> { open(clientX, clientY, items), close() }
//   items: [{label, action, disabled?} | null] -> void  — a null entry renders as a separator.
//   A plain absolutely-positioned <div class="ctx-menu" role="menu"> appended to document.body,
//   each item a real <button type="button" role="menuitem">. Flips left/up when it would overflow
//   the viewport, then clamps so it can never start off-screen at the top/left either.
//   Dismissed on Escape, any outside pointerdown, scroll (capture-phase — 'scroll' does not bubble,
//   so this is the only way a single listener catches it from any scrollable descendant), and
//   after any (non-disabled) item's action runs. Every listener this adds while open is removed on
//   close — nothing outlives one menu's lifetime, so a dismissed-and-reopened menu never
//   accumulates document-level listeners.

const VIEWPORT_MARGIN = 8; // never let the menu touch the very edge of the viewport

export function createContextMenu() {
  let menuEl = null;
  let cleanup = null;

  function close() {
    if (!menuEl) return;
    if (cleanup) { cleanup(); cleanup = null; }
    menuEl.remove();
    menuEl = null;
  }

  function buildItemEl(item) {
    if (item === null) {
      const sep = document.createElement('div');
      sep.className = 'ctx-sep';
      sep.setAttribute('role', 'separator');
      return sep;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.setAttribute('role', 'menuitem');
    btn.className = 'ctx-item';
    btn.textContent = item.label;
    if (item.disabled) {
      btn.disabled = true;
    } else {
      btn.addEventListener('click', () => {
        // Close BEFORE running the action: the action's own runOp may re-render the canvas
        // synchronously (store.subscribe), and the menu — a sibling of the canvas under
        // document.body, not inside it — must never be left dangling regardless of what the
        // action does or whether it errors (errors are toasted by runOp, never thrown here).
        close();
        item.action();
      });
    }
    return btn;
  }

  // open(clientX, clientY, items): clientX/clientY are viewport coordinates (as delivered by the
  // triggering contextmenu event) — the menu is `position: fixed` in css/app.css to match.
  function open(clientX, clientY, items) {
    close(); // only one menu at a time; opening a second closes the first

    const menu = document.createElement('div');
    menu.className = 'ctx-menu';
    menu.setAttribute('role', 'menu');
    for (const item of items) menu.appendChild(buildItemEl(item));

    // Measure before placing: a menu can't know its own rendered width/height until it's in the
    // DOM, and the flip decision below needs both.
    menu.style.visibility = 'hidden';
    document.body.appendChild(menu);
    const { width, height } = menu.getBoundingClientRect();
    const maxX = window.innerWidth - VIEWPORT_MARGIN;
    const maxY = window.innerHeight - VIEWPORT_MARGIN;
    const x = clientX + width > maxX ? clientX - width : clientX;
    const y = clientY + height > maxY ? clientY - height : clientY;
    menu.style.left = `${Math.max(VIEWPORT_MARGIN, x)}px`;
    menu.style.top = `${Math.max(VIEWPORT_MARGIN, y)}px`;
    menu.style.visibility = '';

    menuEl = menu;

    // Capture-phase, deliberately: index.js's own document keydown handler (Escape cancels a
    // rename/gesture, else deselects) was registered long before any menu ever opens, so a
    // same-phase (bubble) listener added here would always run AFTER it — too late to stop it.
    // Capture runs first and stopPropagation() here cuts the event off before it can reach that
    // bubble-phase handler at all (this is the one thing the brief calls out explicitly: Escape
    // closing the menu must not also cancel a rename/gesture or deselect underneath it).
    const onKeyDown = (e) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      e.stopPropagation();
      close();
    };
    // Also capture-phase, so an outside right-click (which itself opens a NEW menu) is detected
    // reliably even if something between here and the target would otherwise stop the event —
    // open() above already closes any existing menu unconditionally, so this is belt-and-suspenders
    // for the "outside pointerdown" requirement specifically, not load-bearing for that case.
    const onPointerDown = (e) => {
      if (!menu.contains(e.target)) close();
    };
    // 'scroll' does not bubble — capture is the only way one listener on window catches it from
    // any scrollable descendant (the results drawer, a dialog body, ...).
    const onScroll = () => close();

    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('scroll', onScroll, true);

    cleanup = () => {
      document.removeEventListener('keydown', onKeyDown, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('scroll', onScroll, true);
    };

    const firstEnabled = menu.querySelector('.ctx-item:not(:disabled)');
    if (firstEnabled) firstEnabled.focus();
  }

  return { open, close };
}
