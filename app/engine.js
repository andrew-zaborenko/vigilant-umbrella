// ===== Yahtzee ("Яхта") advisor engine =====
// Mirrors solver.py exactly: same scoring rules, same expectimax recursion,
// just computed on-demand for the current turn instead of for all 8192 masks.

const CAT_NAMES = ["Единицы","Двойки","Тройки","Четверки","Пятерки","Шестерки",
  "Тройка/Сет","Каре","Фулл-Хаус","Малый стрит","Большой стрит","Яхта","Шанс"];
const IS_UPPER = [true,true,true,true,true,true,false,false,false,false,false,false,false];
const N_CAT = 13;

// ---- 252 canonical dice states (counts per face, sum=5) ----
const STATES = [];
(function gen(remaining, facesLeft, current) {
  if (facesLeft === 1) { STATES.push(current.concat([remaining])); return; }
  for (let k = 0; k <= remaining; k++) gen(remaining - k, facesLeft - 1, current.concat([k]));
})(5, 6, []);
const NS = STATES.length; // 252
const STATE_INDEX = new Map();
STATES.forEach((s, i) => STATE_INDEX.set(s.join(','), i));

function countsOf(dice) { // dice: array of 5 ints 1..6
  const c = [0,0,0,0,0,0];
  for (const d of dice) c[d-1]++;
  return c;
}
function idxOf(counts) { return STATE_INDEX.get(counts.join(',')); }

// ---- reroll distributions: dist[r] = Map(counts-key -> probability) for r fresh dice ----
function buildRerollDists(maxr) {
  const dists = [new Map([["0,0,0,0,0,0", 1.0]])];
  for (let r = 1; r <= maxr; r++) {
    const prev = dists[r-1];
    const cur = new Map();
    for (const [key, p] of prev.entries()) {
      const counts = key.split(',').map(Number);
      for (let f = 0; f < 6; f++) {
        const nc = counts.slice(); nc[f]++;
        const nk = nc.join(',');
        cur.set(nk, (cur.get(nk) || 0) + p / 6);
      }
    }
    dists.push(cur);
  }
  return dists;
}
const REROLL_DISTS = buildRerollDists(5);
const DIST5 = new Float64Array(NS);
for (const [key, p] of REROLL_DISTS[5].entries()) {
  DIST5[idxOf(key.split(',').map(Number))] = p;
}

// ---- scoring, vectorized over the 252 states (computed once at load) ----
function scoreUpper(faceIdx) {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) v[i] = STATES[i][faceIdx] * (faceIdx + 1);
  return v;
}
function scoreThreeKind() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    let best = 0;
    for (let f = 0; f < 6; f++) if (STATES[i][f] >= 3) best = Math.max(best, 3*(f+1));
    v[i] = best;
  }
  return v;
}
function scoreFourKind() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    let best = 0;
    for (let f = 0; f < 6; f++) if (STATES[i][f] >= 4) best = Math.max(best, 4*(f+1));
    v[i] = best;
  }
  return v;
}
function scoreFullHouse() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    const nz = STATES[i].filter(c => c > 0).sort((a,b)=>a-b);
    if (nz.length === 2 && nz[0] === 2 && nz[1] === 3) v[i] = 25;
  }
  return v;
}
function scoreSmallStraight() {
  const seqs = [[1,2,3,4],[2,3,4,5],[3,4,5,6]];
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    const present = new Set();
    STATES[i].forEach((c,f) => { if (c>0) present.add(f+1); });
    if (seqs.some(seq => seq.every(x => present.has(x)))) v[i] = 30;
  }
  return v;
}
function scoreLargeStraight() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    const present = new Set();
    STATES[i].forEach((c,f) => { if (c>0) present.add(f+1); });
    const a = [1,2,3,4,5].every(x=>present.has(x));
    const b = [2,3,4,5,6].every(x=>present.has(x));
    if (a || b) v[i] = 40;
  }
  return v;
}
function scoreYahtzee() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) if (STATES[i].some(c => c === 5)) v[i] = 50;
  return v;
}
function scoreChance() {
  const v = new Float64Array(NS);
  for (let i = 0; i < NS; i++) v[i] = STATES[i].reduce((acc,c,f)=>acc + c*(f+1), 0);
  return v;
}
const CAT_SCORE = [
  scoreUpper(0), scoreUpper(1), scoreUpper(2), scoreUpper(3), scoreUpper(4), scoreUpper(5),
  scoreThreeKind(), scoreFourKind(), scoreFullHouse(),
  scoreSmallStraight(), scoreLargeStraight(), scoreYahtzee(), scoreChance()
];

// ---- EV table loaded from ev_table.bin: Float32Array, shape (8192,64), row-major ----
let EV = null; // Float32Array
async function loadEV(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  EV = new Float32Array(buf);
}
function evAt(mask, s) { return EV[mask * 64 + s]; }

// ---- rr0: for every dice state, best (category, value) at "must score now" ----
function computeRR0(mask, s, openCats) {
  const rr0 = new Float64Array(NS).fill(-Infinity);
  const rr0Cat = new Int8Array(NS).fill(-1);
  for (const c of openCats) {
    const base = CAT_SCORE[c];
    const mask2 = mask | (1 << c);
    for (let i = 0; i < NS; i++) {
      let val;
      if (IS_UPPER[c]) {
        const raw = base[i] + s;
        const bonus = (s < 63 && raw >= 63) ? 35 : 0;
        const newS = Math.min(raw, 63);
        val = base[i] + bonus + evAt(mask2, newS);
      } else {
        val = base[i] + evAt(mask2, s);
      }
      if (val > rr0[i]) { rr0[i] = val; rr0Cat[i] = c; }
    }
  }
  return { rr0, rr0Cat };
}

// enumerate all keep-subsets for a given counts vector, return list of {keepCounts, r}
function keepSubsets(counts) {
  const subsets = [];
  function rec(f, keep) {
    if (f === 6) { subsets.push(keep.slice()); return; }
    for (let k = 0; k <= counts[f]; k++) { keep.push(k); rec(f+1, keep); keep.pop(); }
  }
  rec(0, []);
  return subsets;
}

// expected value of keeping `keep` counts and rerolling the rest, evaluated against `valueArr` (length NS)
function expectedValueOfKeep(keep, valueArr) {
  const r = 5 - keep.reduce((a,b)=>a+b,0);
  const dist = REROLL_DISTS[r];
  let ev = 0;
  for (const [key, p] of dist.entries()) {
    const extra = key.split(',').map(Number);
    const outcome = keep.map((k,f)=>k+extra[f]);
    ev += p * valueArr[idxOf(outcome)];
  }
  return ev;
}

// best keep-action for one specific state, evaluated against valueArr
function bestKeepForState(counts, valueArr) {
  let best = -Infinity, bestKeep = null;
  for (const keep of keepSubsets(counts)) {
    const ev = expectedValueOfKeep(keep, valueArr);
    if (ev > best) { best = ev; bestKeep = keep; }
  }
  return { value: best, keep: bestKeep };
}

// rr1 over ALL 252 states (needed only when 2 rerolls remain)
function computeRR1(rr0) {
  const rr1 = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    rr1[i] = bestKeepForState(STATES[i], rr0).value;
  }
  return rr1;
}

/**
 * Main entry point.
 * mask: bitmask, bit c = 1 means category c is already filled
 * upperSum: current upper-section running total, capped at 63
 * dice: array of 5 ints (1..6), the dice currently showing
 * rerollsRemaining: 2 (just rolled, 2 rerolls left), 1, or 0 (must score)
 */
function suggest(mask, upperSum, dice, rerollsRemaining) {
  const s = Math.min(upperSum, 63);
  const openCats = [];
  for (let c = 0; c < N_CAT; c++) if (!(mask & (1 << c))) openCats.push(c);
  const { rr0, rr0Cat } = computeRR0(mask, s, openCats);
  const counts = countsOf(dice);
  const curIdx = idxOf(counts);

  if (rerollsRemaining === 0 || openCats.length === 0) {
    const cat = rr0Cat[curIdx];
    return { action: 'score', category: cat, categoryName: CAT_NAMES[cat],
             scoreValue: CAT_SCORE[cat][curIdx], expectedFutureValue: rr0[curIdx] };
  }

  const targetArr = (rerollsRemaining === 1) ? rr0 : computeRR1(rr0);
  const { value, keep } = bestKeepForState(counts, targetArr);
  // figure out how many of each face to keep vs reroll, for display
  const keepFaces = [];
  for (let f = 0; f < 6; f++) for (let k = 0; k < keep[f]; k++) keepFaces.push(f+1);
  const rerollCount = 5 - keepFaces.length;
  return { action: 'reroll', keepFaces, rerollCount, expectedValue: value };
}
