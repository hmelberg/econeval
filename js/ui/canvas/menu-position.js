// Pure viewport flip/clamp maths for the context menu — extracted from context-menu.js (task-8
// review, Finding 1) so this is a tested DOM-free module rather than untested logic living inline
// inside a DOM module. No document/window access here, not even at module top level — every input
// the caller already had to read from the DOM (getBoundingClientRect, innerWidth/innerHeight) is
// passed in as a plain number.
//
// This is viewport/CSS-pixel maths (clientX/clientY, innerWidth/innerHeight), not SVG user-space
// node-shape maths — geometry.js's job (see its own header) — so it gets its own file rather than
// folding into geometry.js: one job per file, per this plan's architecture principle.

// clampMenuPosition(clientX, clientY, width, height, innerWidth, innerHeight, margin) -> {left, top}
//   clientX/clientY: the triggering contextmenu event's viewport coordinates (where the menu would
//     naturally open, top-left-anchored).
//   width/height: the menu's own rendered size (getBoundingClientRect(), measured while hidden).
//   innerWidth/innerHeight: the viewport's size.
//   margin: never let the menu touch the very edge of the viewport (also the flip-back clamp floor).
// Flips left/up when the un-flipped placement would overflow the right/bottom edge, then clamps the
// (possibly flipped) result to `margin` so it can never start off-screen at the top/left either —
// this is the one path that matters when the menu itself is wider/taller than the viewport, or the
// click was close enough to the origin that flipping alone would push it negative.
export function clampMenuPosition(clientX, clientY, width, height, innerWidth, innerHeight, margin) {
  const maxX = innerWidth - margin;
  const maxY = innerHeight - margin;
  const x = clientX + width > maxX ? clientX - width : clientX;
  const y = clientY + height > maxY ? clientY - height : clientY;
  return { left: Math.max(margin, x), top: Math.max(margin, y) };
}
