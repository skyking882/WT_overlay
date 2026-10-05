#!/usr/bin/env python3
"""Live loss curves for train_surrogate.py logs (local web page, refreshes every 5 s).

    python3 scripts/train_monitor.py outputs/surrogate_runs/pl12_mlp_e80x5.log --port 8765

Stdlib only; the page loads Chart.js from a CDN.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re

LINE = re.compile(r"^\s+(launch|escape)\[(\d+)\] epoch\s+(\d+)/(\d+)\s+train loss ([\d.]+)"
                  r"(?:\s+val loss ([\d.]+)\s+val acc ([\d.]+)%)?\s+lr ([\d.e+-]+)")


def parse(path):
    runs = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        net, seed, epoch, total, train, val, acc, lr = m.groups()
        run = runs.setdefault(f"{net}[{seed}]", dict(net=net, seed=int(seed), total=int(total), rows=[]))
        run["rows"].append(dict(epoch=int(epoch), train=float(train), val=float(val) if val else None,
                                acc=float(acc) if acc else None, lr=float(lr)))
    return runs


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Curves</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface-1:#1a1a19;--surface-2:#242423;
--text-primary:#fff;--text-secondary:#c3c2b7;--grid:#34342f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;}}
:root[data-theme="dark"]{--surface-1:#1a1a19;--surface-2:#242423;--text-primary:#fff;--text-secondary:#c3c2b7;
--grid:#34342f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;}
body{margin:0;background:var(--surface-1);color:var(--text-primary);font:14px/1.4 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--text-secondary);margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--surface-2);border-radius:8px;padding:12px}
.card h2{font-size:14px;margin:0 0 2px} .card p{color:var(--text-secondary);margin:0 0 8px;font-size:12px}
.box{position:relative;height:260px}
</style></head><body><main>
<h1>训练曲线</h1><p class="sub" id="status">读取中…</p>
<div class="grid">
<div class="card"><h2>规避网络：留出数据损失</h2><p>每个随机种子一条线；越低越好</p><div class="box"><canvas id="eloss"></canvas></div></div>
<div class="card"><h2>规避网络：留出数据准确率</h2><p>判断“能否逃掉”的正确率</p><div class="box"><canvas id="eacc"></canvas></div></div>
<div class="card"><h2>当前种子：训练 vs 留出损失</h2><p>两线分开越多，过拟合越明显</p><div class="box"><canvas id="cur"></canvas></div></div>
<div class="card"><h2>发射网络：留出数据损失</h2><p>判断“不规避是否命中”与飞行时间</p><div class="box"><canvas id="lloss"></canvas></div></div>
</div></main>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SLOTS=['--s1','--s2','--s3','--s4','--s5'];
const charts={};
function chart(id,datasets,ylabel,pct){
  const opts={animation:false,responsive:true,maintainAspectRatio:false,parsing:false,
    interaction:{mode:'nearest',axis:'x',intersect:false},
    plugins:{legend:{labels:{color:css('--text-secondary'),boxWidth:12}},
      tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${pct?c.parsed.y.toFixed(2)+'%':c.parsed.y.toFixed(4)}`}}},
    scales:{x:{type:'linear',title:{display:true,text:'epoch',color:css('--text-secondary')},
        ticks:{color:css('--text-secondary')},grid:{color:css('--grid')}},
      y:{title:{display:true,text:ylabel,color:css('--text-secondary')},ticks:{color:css('--text-secondary')},
        grid:{color:css('--grid')}}}};
  if(charts[id]){charts[id].data.datasets=datasets;charts[id].update('none');return}
  charts[id]=new Chart(document.getElementById(id),{type:'line',data:{datasets},options:opts});
}
const line=(label,color,pts,dash)=>({label,data:pts,borderColor:color,backgroundColor:color,borderWidth:2,
  pointRadius:0,pointHoverRadius:4,borderDash:dash||[],tension:0});
async function refresh(){
  let runs;
  try{runs=await (await fetch('/data')).json()}catch(e){document.getElementById('status').textContent='读取失败，5 秒后重试';return}
  const esc=Object.values(runs).filter(r=>r.net==='escape').sort((a,b)=>a.seed-b.seed);
  const lau=Object.values(runs).filter(r=>r.net==='launch').sort((a,b)=>a.seed-b.seed);
  const col=i=>css(SLOTS[i%SLOTS.length]);
  chart('eloss',esc.map((r,i)=>line(`种子 ${r.seed}`,col(i),r.rows.filter(x=>x.val!=null).map(x=>({x:x.epoch,y:x.val})))),'loss');
  chart('eacc',esc.map((r,i)=>line(`种子 ${r.seed}`,col(i),r.rows.filter(x=>x.acc!=null).map(x=>({x:x.epoch,y:x.acc})))),'%',true);
  chart('lloss',lau.map((r,i)=>line(`种子 ${r.seed}`,col(i),r.rows.filter(x=>x.val!=null).map(x=>({x:x.epoch,y:x.val})))),'loss');
  const all=Object.values(runs); const cur=all[all.length-1];
  if(cur){chart('cur',[line('训练',css('--s1'),cur.rows.map(x=>({x:x.epoch,y:x.train}))),
    line('留出',css('--s2'),cur.rows.filter(x=>x.val!=null).map(x=>({x:x.epoch,y:x.val})),[6,4])],'loss');
    const last=cur.rows[cur.rows.length-1];
    document.getElementById('status').textContent=`正在训练 ${cur.net==='escape'?'规避':'发射'}网络 · 种子 ${cur.seed} · 第 ${last.epoch}/${cur.total} 轮`
      +(last.val!=null?` · 留出损失 ${last.val.toFixed(4)} · 准确率 ${last.acc.toFixed(2)}%`:'')+` · 每 5 秒刷新`;}
}
refresh(); setInterval(refresh,5000);
</script></body></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body, kind = (json.dumps(parse(args.log)).encode(), "application/json") if self.path.startswith("/data") \
                else (PAGE.encode(), "text/html; charset=utf-8")
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"serving {args.log} on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
