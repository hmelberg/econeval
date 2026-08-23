import { mean as distMean, sample as distSample, DIST_NAMES } from './dist.js';

export class ExprError extends Error {
  constructor(message, pos) { super(message); this.pos = pos; }
}

const TOKEN = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(<=|>=|==|!=|[-+*/^()<>,]))/y;

function tokenize(src) {
  const out = [];
  let pos = 0;
  while (pos < src.length) {
    TOKEN.lastIndex = pos;
    const m = TOKEN.exec(src);
    if (!m || m.index !== pos) {
      if (/^\s*$/.test(src.slice(pos))) break;
      throw new ExprError(`unexpected character '${src[TOKEN.lastIndex ?? pos]?.trim() ?? src[pos]}'`, pos);
    }
    const tpos = pos + m[0].length - (m[1] ?? m[2] ?? m[3]).length;
    if (m[1] !== undefined) out.push({ type: 'num', value: Number(m[1]), pos: tpos });
    else if (m[2] !== undefined) out.push({ type: 'name', value: m[2], pos: tpos });
    else out.push({ type: m[3], pos: tpos });
    pos = TOKEN.lastIndex;
  }
  out.push({ type: 'end', pos: src.length });
  return out;
}

const BINARY = { '<': 1, '>': 1, '<=': 1, '>=': 1, '==': 1, '!=': 1, '+': 2, '-': 2, '*': 3, '/': 3, '^': 5 };

function parseTokens(toks) {
  let i = 0;
  const peek = () => toks[i], next = () => toks[i++];
  const expect = (t) => { const tok = next(); if (tok.type !== t) throw new ExprError(`expected '${t}'`, tok.pos); return tok; };

  function parseExpr(minBp) {
    let lhs = parsePrefix();
    for (;;) {
      const op = peek().type;
      const bp = BINARY[op];
      if (bp === undefined || bp < minBp) break;
      next();
      const rhs = parseExpr(op === '^' ? bp : bp + 1);   // ^ right-assoc
      lhs = { kind: 'bin', op, lhs, rhs };
    }
    return lhs;
  }

  function parsePrefix() {
    const tok = next();
    if (tok.type === 'num') return { kind: 'num', value: tok.value };
    if (tok.type === '-') return { kind: 'neg', arg: parseExpr(4) };  // binds looser than ^
    if (tok.type === '(') { const e = parseExpr(0); expect(')'); return e; }
    if (tok.type === 'name') {
      if (peek().type === '(') {
        next();
        const args = [];
        if (peek().type !== ')') { args.push(parseExpr(0)); while (peek().type === ',') { next(); args.push(parseExpr(0)); } }
        expect(')');
        return { kind: 'call', name: tok.value, args, pos: tok.pos };
      }
      return { kind: 'name', name: tok.value, pos: tok.pos };
    }
    throw new ExprError(`unexpected '${tok.type}'`, tok.pos);
  }

  const ast = parseExpr(0);
  if (peek().type !== 'end') throw new ExprError(`unexpected '${peek().type}'`, peek().pos);
  return ast;
}

function evalAst(ast, env) {
  switch (ast.kind) {
    case 'num': return ast.value;
    case 'neg': return -evalAst(ast.arg, env);
    case 'name':
      if (ast.name === 'rest') throw new ExprError('rest is only valid as a whole transition/branch probability', ast.pos);
      return env.get(ast.name);
    case 'bin': {
      const a = evalAst(ast.lhs, env), b = evalAst(ast.rhs, env);
      switch (ast.op) {
        case '+': return a + b; case '-': return a - b; case '*': return a * b;
        case '/': return a / b; case '^': return Math.pow(a, b);
        case '<': return a < b ? 1 : 0; case '>': return a > b ? 1 : 0;
        case '<=': return a <= b ? 1 : 0; case '>=': return a >= b ? 1 : 0;
        case '==': return a === b ? 1 : 0; case '!=': return a !== b ? 1 : 0;
      }
    }
    case 'call': return evalCall(ast, env);
  }
}

function evalCall(ast, env) {
  const { name, args } = ast;
  if (DIST_NAMES.has(name)) {
    try {
      const d = { name, args: args.map(a => evalAst(a, env)) };
      return env.mode === 'sample' ? distSample(d, env.rand) : distMean(d);
    } catch (e) {
      throw new ExprError(e.message, ast.pos);
    }
  }
  switch (name) {
    case 'min': return Math.min(...args.map(a => evalAst(a, env)));
    case 'max': return Math.max(...args.map(a => evalAst(a, env)));
    case 'if': return evalAst(args[0], env) !== 0 ? evalAst(args[1], env) : evalAst(args[2], env);
    case 'rate_to_prob': return 1 - Math.exp(-evalAst(args[0], env) * env.cycleYears);
    case 'prob_to_rate': return -Math.log(1 - evalAst(args[0], env)) / env.cycleYears;
    case 'rescale_prob': {
      const p = evalAst(args[0], env), years = evalAst(args[1], env);
      return 1 - Math.pow(1 - p, env.cycleYears / years);
    }
    case 'lookup': {
      const tname = args[0]; // identifier arg
      if (tname.kind !== 'name') throw new ExprError('lookup: first argument must be a table name', ast.pos);
      const table = env.tables?.[tname.name];
      if (!table) throw new ExprError(`lookup: unknown table '${tname.name}'`, ast.pos);
      const cols = Object.keys(table);
      const colName = args[2] ? (args[2].kind === 'name' ? args[2].name : null) : cols[1];
      if (!colName || !table[colName]) throw new ExprError(`lookup: unknown column`, ast.pos);
      const xs = table[cols[0]], ys = table[colName];
      const x = evalAst(args[1], env);
      if (x <= xs[0]) return ys[0];
      if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
      let i = 1; while (xs[i] < x) i++;
      const f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + f * (ys[i] - ys[i - 1]);
    }
    default: throw new ExprError(`unknown function: ${name}`, ast.pos);
  }
}

export function compile(src) {
  if (typeof src === 'number') return { src, names: new Set(), eval: () => src };
  const ast = parseTokens(tokenize(String(src)));
  const names = new Set();
  (function walk(a) {
    if (a.kind === 'name' && a.name !== 'rest') names.add(a.name);
    else if (a.kind === 'neg') walk(a.arg);
    else if (a.kind === 'bin') { walk(a.lhs); walk(a.rhs); }
    else if (a.kind === 'call') {
      // lookup's table/column args are identifiers, not env names — skip them
      a.args.forEach((w, i) => { if (!(a.name === 'lookup' && (i === 0 || i === 2))) walk(w); });
    }
  })(ast);
  return { src: String(src), names, eval: (env) => evalAst(ast, env) };
}
