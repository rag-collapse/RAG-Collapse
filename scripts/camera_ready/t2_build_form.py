"""Build a self-contained offline HTML annotation form for T2 from t2_sample.jsonl.

Opens with file:// (no server). Data is embedded; answers autosave to localStorage; an
Export button downloads the annotations as JSON, which scripts/camera_ready/t2_score.py scores.

Per item the annotator: (1) marks each extracted entity ✓ (a real entity, present in the answer)
or ✗ (spurious / mis-extracted); (2) lists entities the extractor MISSED (comma-separated).
Precision = ✓ / extracted; Recall = ✓ / (✓ + missed); F1 the harmonic mean.
"""
import json, os, html

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T2")
items = [json.loads(l) for l in open(os.path.join(OUT, "t2_sample.jsonl"), encoding="utf-8")]

DATA = json.dumps(items, ensure_ascii=False)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>T2 entity-extraction agreement</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header b{font-size:15px} .sp{flex:1}
 button{font:inherit;padding:6px 12px;border:1px solid #888;border-radius:6px;background:#fff;cursor:pointer}
 button.primary{background:#1f6feb;color:#fff;border-color:#1f6feb}
 #wrap{max-width:900px;margin:0 auto;padding:16px}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:16px;margin:0 0 16px}
 .meta{color:#666;font-size:12px;margin-bottom:8px}
 .q{font-weight:600;margin-bottom:6px}
 .ans{background:#f0f3f7;border-radius:6px;padding:10px;margin:8px 0;white-space:pre-wrap}
 .ents{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
 .ent{border:1px solid #ccc;border-radius:20px;padding:4px 10px;cursor:pointer;user-select:none;background:#fafafa}
 .ent.ok{background:#e6ffed;border-color:#34a853} .ent.bad{background:#ffeaea;border-color:#d93025;text-decoration:line-through}
 .missed{width:100%;box-sizing:border-box;padding:8px;border:1px solid #ccc;border-radius:6px;font:inherit}
 label.small{font-size:12px;color:#555;display:block;margin:8px 0 2px}
 .done{outline:2px solid #34a85355}
 .prog{font-variant-numeric:tabular-nums}
</style></head><body>
<header>
 <b>T2 — entity-extraction agreement</b>
 <span class="prog" id="prog"></span>
 <span class="sp"></span>
 <button onclick="exportJSON()" class="primary">Export annotations (JSON)</button>
 <button onclick="if(confirm('Clear all saved annotations on this browser?')){localStorage.clear();location.reload()}">Reset</button>
</header>
<div id="wrap">
 <p style="color:#555">For each answer: click each extracted entity to toggle <b style="color:#34a853">✓ correct</b> (a real
 entity actually stated in the answer) vs <b style="color:#d93025">✗ wrong</b> (spurious, duplicate, or not in the
 answer). In the box, type any entities the extractor <b>missed</b>, comma-separated. Progress saves automatically;
 click <b>Export</b> when done and hand the JSON to whoever runs <code>t2_score.py</code>.</p>
 <div id="items"></div>
</div>
<script>
const DATA = __DATA__;
const KEY = "t2_annotations_v1";
let store = {};
try{ store = JSON.parse(localStorage.getItem(KEY)||"{}"); }catch(e){ store = {}; }
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(store)); }catch(e){} prog(); }
function get(id){ return store[id] || (store[id]={marks:{},missed:""}); }
function prog(){
  const done = DATA.filter(it=>{const a=store[it.item_id]; return a && Object.keys(a.marks||{}).length>0;}).length;
  document.getElementById("prog").textContent = done+" / "+DATA.length+" annotated";
}
function render(){
  const root = document.getElementById("items");
  DATA.forEach(it=>{
    const a = get(it.item_id);
    const card = document.createElement("div"); card.className="card"; card.id="c_"+it.item_id;
    const ents = it.extracted_entities.map((e,i)=>{
      const st = a.marks[i]; const cls = st==="ok"?"ent ok":st==="bad"?"ent bad":"ent";
      return `<span class="${cls}" data-id="${it.item_id}" data-i="${i}">${escapeHtml(e)}</span>`;
    }).join("") || '<i style="color:#999">(extractor returned no entities)</i>';
    card.innerHTML = `<div class="meta">${it.item_id} · ${it.source} · round ${it.round} · run ${it.run_id}</div>
      <div class="q">Q: ${escapeHtml(it.question_text)}</div>
      <div class="ans">${escapeHtml(it.answer)}</div>
      <label class="small">Extracted entities — click to mark ✓/✗:</label>
      <div class="ents">${ents}</div>
      <label class="small">Missed entities (present in the answer but not extracted), comma-separated:</label>
      <input class="missed" data-id="${it.item_id}" placeholder="e.g. Katie Feeney, Vanessa Lopes" value="${escapeHtml(a.missed||"")}">`;
    root.appendChild(card);
  });
  root.addEventListener("click", e=>{
    const t = e.target; if(!t.classList.contains("ent")) return;
    const a = get(t.dataset.id); const i = t.dataset.i;
    const cur = a.marks[i]; a.marks[i] = cur==="ok"?"bad":cur==="bad"?undefined:"ok";
    if(a.marks[i]===undefined) delete a.marks[i];
    t.className = "ent" + (a.marks[i]==="ok"?" ok":a.marks[i]==="bad"?" bad":"");
    save();
  });
  root.addEventListener("input", e=>{
    if(!e.target.classList.contains("missed")) return;
    get(e.target.dataset.id).missed = e.target.value; save();
  });
  prog();
}
function escapeHtml(s){return (s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function exportJSON(){
  const out = {annotated_at:new Date().toISOString(), n_items:DATA.length, annotations:store};
  const blob = new Blob([JSON.stringify(out,null,1)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="t2_annotations.json"; a.click();
}
render();
</script></body></html>"""

out = os.path.join(OUT, "t2_annotation.html")
open(out, "w", encoding="utf-8").write(PAGE.replace("__DATA__", DATA))
print("wrote", out, "with", len(items), "items")
