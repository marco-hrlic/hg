"""Render graph-adlc graphs as one self-contained, offline HTML page.

    gadlc-viz evaluation/llm_graph/out/live -o graphs.html                  # every project/arm in a run
    gadlc-viz out/live/toolshare/G1/trace.json --oracle evaluation/llm_graph/oracle
    gadlc export --out snap.json && gadlc-viz snap.json -o snap.html         # a plain engine snapshot

Inputs may be a run directory (``<project>/<arm>/trace.json`` below it), a replay ``trace.json`` or a
``gadlc export`` snapshot. Layout is computed here, deterministically, on the union of all graphs of a
project, so a node keeps its position when you switch arms. The page has no external dependencies.

Overlays, when the trace carries them: the probe origin, the engine's predicted impact targets and, with
``--oracle``, the labelled impact (hit / false positive / missed / critical missed). Any graph can be
compared with another of the same project: what only the other one has is drawn as faint ghosts.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

KIND_ORDER = ("goal", "capability", "requirement", "question", "contract", "component", "test", "artifact")
ARM_ORDER = {"D": 0, "G0": 1, "G1": 2, "G2": 3, "H": 4}
COL_W, ROW_H, NODE_W, NODE_H, PAD, LABEL_W = 236, 112, 208, 60, 28, 104
PER_LINE, LINE_H = 6, 80  # a kind with more nodes than PER_LINE wraps onto extra lines, LINE_H apart


# -- loading -------------------------------------------------------------------------------------------
def _graph_entry(label: str, project: str, arm: str, snap: dict, trace: dict | None) -> dict:
    entry: dict[str, Any] = {"label": label, "project": project, "arm": arm,
                             "graph_revision": snap.get("graph_revision"),
                             "nodes": snap.get("nodes", []), "edges": snap.get("edges", [])}
    if trace:
        client = trace.get("client") or {}
        entry["client"] = {k: client.get(k) for k in ("provider", "model", "is_fake") if k in client}
        probe = trace.get("probe") or {}
        pred = (trace.get("prediction") or {}).get(probe.get("event", ""), {}) if probe else {}
        if probe:
            entry["probe"] = {"event": probe.get("event"), "origin": probe.get("node_id"),
                              "aspect": probe.get("aspect"), "description": probe.get("description"),
                              "targets": pred.get("targets", []), "pairs": pred.get("pairs", []),
                              "pruned": pred.get("pruned", []), "provisional": pred.get("provisional", [])}
    return entry


def _find_reference_dir(run_file: Path) -> Path | None:
    for parent in run_file.resolve().parents:  # evaluation/llm_graph/out/<mode>/<project>/<arm>/trace.json
        if parent.name == "out" and (parent.parent / "reference").is_dir():
            return parent.parent / "reference"
    return None


def load_inputs(paths: list[str], reference: str | None = None) -> tuple[list[dict], list[str]]:
    """Return (graph entries, notes on skipped inputs) from run dirs, trace files and snapshots.

    Arm D has no graph and is skipped. Arm H traces store no graph (it is the hand-curated seed), so its
    graph is read from ``reference`` or, by default, the ``reference/`` directory next to ``out/``.
    """
    out: list[dict] = []
    notes: list[str] = []
    for raw in paths:
        p = Path(raw)
        files = sorted(p.glob("*/*/trace.json")) if p.is_dir() else [p]
        if p.is_dir() and not files:
            raise SystemExit(f"{p}: no <project>/<arm>/trace.json below it")
        for f in files:
            doc = json.loads(f.read_text())
            if "nodes" in doc and "edges" in doc and "probe" not in doc:
                out.append(_graph_entry(f.stem, f.stem, "snapshot", doc, None))
                continue
            if "arm" not in doc and "final_graph" not in doc:
                raise SystemExit(f"{f}: neither a replay trace nor a graph snapshot")
            project = doc.get("project") or f.parent.parent.name
            arm = doc.get("arm") or f.parent.name
            snap = doc.get("final_graph")
            if not snap and arm == "H":
                rdir = Path(reference) if reference else _find_reference_dir(f)
                rf = rdir / f"{project}.json" if rdir else None
                if rf and rf.is_file():
                    snap = json.loads(rf.read_text())
                else:
                    notes.append(f"{project}/H: reference graph not found (pass --reference)")
                    continue
            if not snap:
                notes.append(f"{project}/{arm}: no graph in trace (arm D reads documents directly)")
                continue
            out.append(_graph_entry(f"{project} / {arm}", project, arm, snap, doc))
    out.sort(key=lambda e: (e["project"], ARM_ORDER.get(e["arm"], 9), e["label"]))
    return out, notes


def load_oracles(path: str | None, projects: set[str]) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    found = {}
    for proj in sorted(projects):
        f = p / f"{proj}.json" if p.is_dir() else p
        if f.is_file():
            doc = json.loads(f.read_text())
            if doc.get("project", proj) == proj:
                found[proj] = {"version": doc.get("oracle_version"), "labeler": doc.get("labeler"),
                               "impact": doc.get("impact", {})}
    return found


# -- layout --------------------------------------------------------------------------------------------
def layout(entries: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Per project: node id -> [x, y]. One row per kind; order within a row by barycentre sweeps."""
    result: dict[str, dict[str, list[float]]] = {}
    for project in sorted({e["project"] for e in entries}):
        kinds: dict[str, str] = {}
        adj: dict[str, set[str]] = {}
        for e in (x for x in entries if x["project"] == project):
            for n in e["nodes"]:
                kinds.setdefault(n["id"], n.get("kind", "?"))
            for ed in e["edges"]:
                adj.setdefault(ed["src"], set()).add(ed["dst"])
                adj.setdefault(ed["dst"], set()).add(ed["src"])
        order = [k for k in KIND_ORDER if k in kinds.values()]
        order += sorted(set(kinds.values()) - set(order))
        cols = {k: sorted(n for n, kk in kinds.items() if kk == k) for k in order}
        pos = {n: float(i) for k in order for i, n in enumerate(cols[k])}
        for _ in range(6):  # barycentre sweeps; ties keep id order, so the result is deterministic
            for k in order:
                def key(n: str) -> tuple[float, str]:
                    nb = [pos[m] for m in adj.get(n, ()) if m in pos and kinds[m] != k]
                    return (sum(nb) / len(nb) if nb else pos[n], n)
                cols[k].sort(key=key)
                for i, n in enumerate(cols[k]):
                    pos[n] = float(i)
        widest = min(max((len(c) for c in cols.values()), default=0), PER_LINE)
        coords, rows, y = {}, [], PAD + 20
        for k in order:
            rows.append([k, y])
            lines = [cols[k][i:i + PER_LINE] for i in range(0, len(cols[k]), PER_LINE)]
            for li, line in enumerate(lines):
                off = (widest - len(line)) * COL_W / 2
                for i, n in enumerate(line):
                    coords[n] = [PAD + LABEL_W + off + i * COL_W, y + li * LINE_H]
            y += (len(lines) - 1) * LINE_H + ROW_H
        result[project] = {"coords": coords, "rows": rows,
                           "width": PAD * 2 + LABEL_W + max(widest - 1, 0) * COL_W + NODE_W,
                           "height": y - ROW_H + NODE_H + PAD}
    return result


# -- page ----------------------------------------------------------------------------------------------
def render(entries: list[dict], oracles: dict[str, dict], title: str = "graph-adlc graphs") -> str:
    data = {"graphs": entries, "layout": layout(entries), "oracles": oracles,
            "geom": {"colW": COL_W, "rowH": ROW_H, "nodeW": NODE_W, "nodeH": NODE_H, "pad": PAD}}
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gadlc-viz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="run dir, trace.json or gadlc export snapshot (any mix)")
    ap.add_argument("-o", "--out", default="graph.html", help="output HTML file (default graph.html)")
    ap.add_argument("--oracle", default=None, help="oracle dir (<project>.json) or file, for the impact overlay")
    ap.add_argument("--reference", default=None,
                    help="dir of hand-curated arm-H graphs (<project>.json); found automatically next to out/")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    entries, notes = load_inputs(a.inputs, a.reference)
    for n in notes:
        print(f"skipped {n}", file=sys.stderr)
    if not entries:
        print("no graphs found in the inputs", file=sys.stderr)
        return 1
    oracles = load_oracles(a.oracle, {e["project"] for e in entries})
    title = a.title or ("graph-adlc: " + ", ".join(dict.fromkeys(e["project"] for e in entries)))
    Path(a.out).write_text(render(entries, oracles, title))
    print(f"wrote {a.out} ({len(entries)} graph{'s' if len(entries) != 1 else ''})")
    return 0


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbfa;--panel:#ffffff;--fg:#1d1d1b;--muted:#6b6b66;--line:#dcdcd6;--edge:#8a8a84;
--goal:#6a4c93;--capability:#3d6fb6;--requirement:#1f8a70;--question:#c2410c;--contract:#b8860b;
--component:#2b6cb0;--test:#7c3aed;--artifact:#71717a;--hit:#15803d;--fp:#b45309;--miss:#dc2626;--origin:#111}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--panel:#1f1f1d;--fg:#ececea;
--muted:#9b9b95;--line:#34342f;--edge:#7c7c76;--origin:#f5f5f4;--goal:#a78bdb;--capability:#7aa7e8;
--requirement:#4cc9a3;--question:#fb923c;--contract:#e0b84a;--component:#6aa6ee;--test:#a78bfa;--artifact:#a1a1aa;
--hit:#4ade80;--fp:#fbbf24;--miss:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0}label{color:var(--muted);font-size:13px;display:flex;gap:6px;align-items:center}
select,input{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
main{display:flex;min-height:calc(100vh - 60px)}#wrap{flex:1;overflow:auto;padding:8px}
aside{width:340px;max-width:40vw;border-left:1px solid var(--line);background:var(--panel);padding:14px 16px;overflow:auto}
@media (max-width:760px){main{flex-direction:column}aside{width:auto;max-width:none;border-left:0;border-top:1px solid var(--line)}}
svg{display:block;width:100%;height:auto;min-width:720px}.n rect{fill:var(--panel);stroke-width:1.5}.n text{fill:var(--fg);font-size:12px;pointer-events:none}
.n{cursor:pointer}.nid{font:10.5px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--muted)}.n.dim,.e.dim{opacity:.12}.ghost{opacity:.35}.ghost rect{stroke-dasharray:3 3}
.e path{fill:none;stroke-width:1.4}.e.hyp path{stroke-dasharray:6 4}.e.del path{stroke-dasharray:2 3}
.e text{font-size:10px;fill:var(--muted)}.colh{font-size:11px;fill:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.ring{fill:none;stroke-width:3}.sel rect{stroke-width:3}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--muted);margin:10px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
h2{font-size:14px;margin:14px 0 6px}dl{margin:0}dt{color:var(--muted);font-size:12px;margin-top:8px}dd{margin:0;word-break:break-word}
.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin:2px 3px 2px 0;font-size:12px}
.warn{color:var(--fp)}pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:6px;border-radius:6px}
table{border-collapse:collapse;font-size:12px;width:100%}td{padding:2px 4px;border-top:1px solid var(--line);vertical-align:top}
</style></head><body>
<header><h1>__TITLE__</h1>
<label>Graph <select id="g"></select></label>
<label>Compare with <select id="cmp"><option value="">none</option></select></label>
<label><input type="checkbox" id="impact" checked> impact overlay</label>
<label><input type="checkbox" id="deleted"> deleted</label>
<label><input type="checkbox" id="elabels"> edge labels</label>
<label>Labels <select id="lab"><option value="both">title + id</option><option value="title">title</option><option value="id">id</option></select></label>
<label>Find <input id="q" size="12" placeholder="id or title"></label>
</header>
<script type="application/json" id="data">__DATA__</script>
<main><div id="wrap"><svg id="svg"></svg></div><aside id="side"></aside></main>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg', G=D.geom;
const REL={refines:'--goal',depends_on:'--component',constrained_by:'--contract',implements:'--requirement',
 verifies:'--test',realized_by:'--artifact',blocks:'--question'};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let sel=null;
function mk(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function pref(k){try{return localStorage.getItem('gadlc-viz-'+k)}catch(e){return null}}
function save(k,v){try{localStorage.setItem('gadlc-viz-'+k,v)}catch(e){}}
D.graphs.forEach((g,i)=>$('g').add(new Option(g.label+(g.client&&g.client.is_fake?' (scripted)':''),i)));
const last=+pref('g'); if(last>=0&&last<D.graphs.length)$('g').value=last;
function fillCmp(){const g=cur(),c=$('cmp'),old=c.value;c.length=1;
 D.graphs.forEach((h,i)=>{if(h!==g&&h.project===g.project)c.add(new Option(h.label,i))});
 c.value=[...c.options].some(o=>o.value===old)?old:''}
function cur(){return D.graphs[+$('g').value]}
function live(g,showDel){return{nodes:g.nodes.filter(n=>showDel||!n.deleted),edges:g.edges.filter(e=>showDel||!e.deleted)}}
function classify(g){ // impact overlay: id -> class
 const out={},p=g.probe;if(!p)return out;const o=(D.oracles[g.project]||{}).impact||{},lab=o[p.event];
 const pred=new Set(p.targets);pred.delete(p.origin);out[p.origin]='origin';
 if(!lab){pred.forEach(t=>out[t]='pred');return out}
 const nec=new Map((lab.necessary||[]).map(x=>[x.target,x.critical])),unk=new Set(lab.unknown||[]);
 pred.forEach(t=>out[t]=nec.has(t)?'hit':unk.has(t)?'pred':'fp');
 nec.forEach((crit,t)=>{if(!pred.has(t)&&t!==p.origin)out[t]=crit?'critmiss':'miss'});return out}
const RING={origin:'--origin',pred:'--muted',hit:'--hit',fp:'--fp',miss:'--miss',critmiss:'--miss'};
function draw(){
 const g=cur(),L=D.layout[g.project],showDel=$('deleted').checked,c=$('cmp').value,o=c?D.graphs[+c]:null;
 const s=$('svg');s.innerHTML='';s.style.maxWidth=L.width+'px';
 s.setAttribute('viewBox',`0 0 ${L.width} ${L.height}`);
 const defs=mk('defs',{},s);
 Object.entries(REL).concat([['ghost','--edge']]).forEach(([r,v])=>{const m=mk('marker',{id:'a-'+r,viewBox:'0 0 10 10',refX:9,refY:5,
  markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},defs);mk('path',{d:'M0,0L10,5L0,10z',fill:`var(${v})`},m)});
 L.rows.forEach(([k,y])=>{const t=mk('text',{x:G.pad,y:y+G.nodeH/2+4,class:'colh'},s);t.textContent=k});
 const mine=live(g,showDel),ids=new Set(mine.nodes.map(n=>n.id)),eids=new Set(mine.edges.map(e=>e.src+'|'+e.relation+'|'+e.dst));
 let nodes=mine.nodes.map(n=>({...n,_g:false})),edges=mine.edges.map(e=>({...e,_g:false}));
 if(o){const t=live(o,showDel);t.nodes.forEach(n=>{if(!ids.has(n.id))nodes.push({...n,_g:true})});
  t.edges.forEach(e=>{if(!eids.has(e.src+'|'+e.relation+'|'+e.dst))edges.push({...e,_g:true})})}
 const at=id=>L.coords[id], eg=mk('g',{},s), ng=mk('g',{},s), cls=$('impact').checked?classify(g):{};
 const nb={};edges.forEach(e=>{(nb[e.src]=nb[e.src]||new Set).add(e.dst);(nb[e.dst]=nb[e.dst]||new Set).add(e.src)});
 edges.forEach(e=>{const a=at(e.src),b=at(e.dst);if(!a||!b)return;
  const same=a[1]===b[1],down=b[1]>a[1],x1=a[0]+G.nodeW/2,x2=b[0]+G.nodeW/2;
  const y1=same||!down?a[1]:a[1]+G.nodeH,y2=same||down?b[1]:b[1]+G.nodeH,dy=same?34:Math.max(30,Math.abs(y2-y1)/2);
  const d=same?`M${x1},${y1}C${x1},${y1-dy} ${x2},${y2-dy} ${x2},${y2}`
   :`M${x1},${y1}C${x1},${y1+(down?dy:-dy)} ${x2},${y2-(down?dy:-dy)} ${x2},${y2}`;
  const gg=mk('g',{class:'e'+(e.status==='hypothesized'?' hyp':'')+(e.deleted?' del':'')+(e._g?' ghost':''),'data-s':e.src,'data-d':e.dst},eg);
  const col=e._g?'--edge':(REL[e.relation]||'--edge');
  const p=mk('path',{d,stroke:`var(${col})`,'marker-end':`url(#a-${e._g?'ghost':(REL[e.relation]?e.relation:'ghost')})`},gg);
  mk('title',{},p).textContent=`${e.src} ${e.relation} ${e.dst}\nstatus: ${e.status}${e.evidence_ref?'\nevidence: '+e.evidence_ref:''}${e._g?'\n(only in '+o.label+')':''}`;
  if($('elabels').checked){const t=mk('text',{x:(x1+x2)/2,y:same?y1-dy*0.75:(y1+y2)/2-3,'text-anchor':'middle'},gg);t.textContent=e.relation}});
 nodes.forEach(n=>{const p=at(n.id);if(!p)return;
  const gg=mk('g',{class:'n'+(n._g?' ghost':'')+(sel===n.id?' sel':''),transform:`translate(${p[0]},${p[1]})`,'data-id':n.id},ng);
  mk('rect',{width:G.nodeW,height:G.nodeH,rx:7,stroke:`var(--${n.kind},var(--edge))`},gg);
  if(cls[n.id]&&!n._g){mk('rect',{class:'ring',x:-4,y:-4,width:G.nodeW+8,height:G.nodeH+8,rx:10,stroke:`var(${RING[cls[n.id]]})`,
   'stroke-dasharray':cls[n.id]==='miss'||cls[n.id]==='critmiss'?'5 3':'none'},gg)}
  if(cls[n.id]==='critmiss'&&!n._g){const t=mk('text',{x:G.nodeW-4,y:-7,'text-anchor':'end',style:'fill:var(--miss);font-weight:600;font-size:11px'},gg);t.textContent='critical miss'}
  label(gg,n);
  gg.addEventListener('mouseenter',()=>focus(n.id,nb));gg.addEventListener('mouseleave',()=>focus(null));
  gg.addEventListener('click',()=>{sel=sel===n.id?null:n.id;draw();side()})});
 const q=$('q').value.trim().toLowerCase();if(q)ng.querySelectorAll('.n').forEach(el=>{
  const n=nodes.find(x=>x.id===el.dataset.id);if(!(n.id+' '+n.title).toLowerCase().includes(q))el.classList.add('dim')});
 side()}
// Node label: wrapped title (up to 2 lines) above a small monospace id; mode from the "Labels" control.
function niceTitle(n){let t=(n.title||'').trim();if(n.kind==='question')t=t.replace(/^Resolve:\s*/i,'');
 return t?t.charAt(0).toUpperCase()+t.slice(1):n.id}
function wrap(el,str,maxW,maxLines,x,y0,lh){const words=str.split(/\s+/);let line=[],n=0,y=y0;
 const span=()=>{const t=mk('tspan',{x,y},el);n++;y+=lh;return t};let t=span();
 for(let i=0;i<words.length;i++){line.push(words[i]);t.textContent=line.join(' ');
  if(t.getComputedTextLength()>maxW&&line.length>1){line.pop();t.textContent=line.join(' ');
   if(n===maxLines){t.textContent=line.join(' ')+' …';fit(t,maxW);return n}
   line=[words[i]];t=span();t.textContent=words[i]}}
 fit(t,maxW);return n}
function fit(t,maxW){while(t.getComputedTextLength()>maxW&&t.textContent.length>2)t.textContent=t.textContent.slice(0,-2)+'…'}
function label(gg,n){const mode=$('lab').value,W=G.nodeW-18,del=n.deleted?' ✕':'';
 mk('title',{},gg).textContent=`${n.id}${del}\n${n.title||''}${n.body&&n.body!==n.title?'\n\n'+n.body:''}`;
 if(mode==='id'){const t=mk('text',{x:9,y:G.nodeH/2+4,style:'font-weight:600'},gg);t.textContent=n.id+del;fit(t,W);return}
 const tt=mk('text',{class:'nt',style:'font-weight:600;font-size:12.5px'},gg);
 const lines=wrap(tt,niceTitle(n)+del,W,2,9,mode==='title'?24:18,15);
 if(mode==='title'&&lines===1)tt.firstChild.setAttribute('y',G.nodeH/2+4);
 if(mode==='both'){const t=mk('text',{x:9,y:G.nodeH-9,class:'nid'},gg);t.textContent=n.id;fit(t,W)}}
function focus(id,nb){document.querySelectorAll('#svg .n').forEach(el=>el.classList.toggle('dim',!!id&&el.dataset.id!==id&&!(nb[id]||new Set).has(el.dataset.id)));
 document.querySelectorAll('#svg .e').forEach(el=>el.classList.toggle('dim',!!id&&el.dataset.s!==id&&el.dataset.d!==id))}
function side(){const g=cur(),a=$('side'),o=$('cmp').value?D.graphs[+$('cmp').value]:null;let h='';
 const kinds=[...new Set(g.nodes.map(n=>n.kind))];
 h+=`<h2>${esc(g.label)}</h2><div>${g.nodes.filter(n=>!n.deleted).length} nodes · ${g.edges.filter(e=>!e.deleted).length} edges · revision ${esc(g.graph_revision)}</div>`;
 if(g.client)h+=`<div>${g.client.is_fake?'<span class="warn">scripted fake model: not evidence</span>':'model: '+esc((g.client.provider||'')+' '+(g.client.model||''))}</div>`;
 h+='<div class="legend">'+kinds.map(k=>`<span><span class="sw" style="border:2px solid var(--${k},var(--edge))"></span>${esc(k)}</span>`).join('')+'</div>';
 h+='<div class="legend"><span>— confirmed</span><span>- - hypothesized</span>'+(o?`<span style="opacity:.5">ghost = only in ${esc(o.label)}</span>`:'')+'</div>';
 const n=sel&&(g.nodes.find(x=>x.id===sel)||(o&&o.nodes.find(x=>x.id===sel)));
 if(n){const src=g.nodes.includes(n)?g:o;h+=`<h2>${esc(niceTitle(n))}</h2><div class="nid" style="font-size:12px;color:var(--muted)">${esc(n.id)}</div><dl><dt>kind</dt><dd>${esc(n.kind)}${src!==g?' (only in '+esc(o.label)+')':''}</dd><dt>title</dt><dd>${esc(n.title)}</dd>`;
  if(n.body)h+=`<dt>body</dt><dd>${esc(n.body)}</dd>`;if(n.data&&Object.keys(n.data).length)h+=`<dt>data</dt><dd><pre>${esc(JSON.stringify(n.data,null,1))}</pre></dd>`;
  h+=`<dt>rev</dt><dd>${esc(n.rev)}</dd></dl><h2>Edges</h2><table>`+src.edges.filter(e=>e.src===n.id||e.dst===n.id)
   .map(e=>`<tr><td>${esc(e.src)}</td><td>${esc(e.relation)}${e.status==='hypothesized'?' (hyp.)':''}${e.deleted?' (deleted)':''}</td><td>${esc(e.dst)}</td></tr>`).join('')+'</table>'}
 else h+='<p style="color:var(--muted)">Click a node for details; hover to highlight its neighbours.</p>';
 const p=g.probe;if(p){const c=classify(g),lab=((D.oracles[g.project]||{}).impact||{})[p.event];
  h+=`<h2>Probe ${esc(p.event)}: ${esc(p.aspect)} change to ${esc(p.origin)}</h2><div>${esc(p.description)}</div>`;
  h+=`<dt>predicted targets</dt><dd>${p.targets.map(t=>`<span class="tag">${esc(t)}</span>`).join('')||'none'}</dd>`;
  if(p.pruned.length)h+=`<dt>pruned</dt><dd>${p.pruned.map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</dd>`;
  if(lab){const by=k=>Object.entries(c).filter(([,v])=>v===k).map(([t])=>`<span class="tag">${esc(t)}</span>`).join('')||'none';
   h+=`<dt><span class="sw" style="background:var(--hit)"></span>hit</dt><dd>${by('hit')}</dd><dt><span class="sw" style="background:var(--fp)"></span>not in labels (false positive)</dt><dd>${by('fp')}</dd>`;
   h+=`<dt><span class="sw" style="background:var(--miss)"></span>missed: critical</dt><dd>${by('critmiss')}</dd><dt>missed: other</dt><dd>${by('miss')}</dd>`;
   const or=D.oracles[g.project];h+=`<p style="color:var(--muted);font-size:12px">Labels: oracle v${esc(or.version)}, ${esc(or.labeler)}.</p>`}
  else h+='<p style="color:var(--muted);font-size:12px">No oracle loaded (pass --oracle) — grey rings are predictions only.</p>'}
 a.innerHTML=h}
$('g').onchange=()=>{save('g',$('g').value);sel=null;fillCmp();draw()};
['cmp','impact','deleted','elabels'].forEach(i=>$(i).onchange=draw);
{const v=pref('lab');if(v)$('lab').value=v}$('lab').onchange=()=>{save('lab',$('lab').value);draw()};$('q').oninput=draw;
fillCmp();draw();
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
