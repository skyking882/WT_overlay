#!/usr/bin/env python3
"""Build resumable straight-target reach/timing tables via missile_sim's public API.

Full grids belong on workstation; use --limit 1 --workers 1 for a local probe.
At most 48 workers. Existing compatible cells are retained; each finished cell
is atomically saved. Runtime estimates use measured seconds per new cell.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wt_overlay.engagement import default_library, LAUNCH_OPTIONS


class StraightTarget:
    def __init__(self, position, velocity):
        self.position,self.velocity=position,velocity

    def state_at(self,t):
        from aim120_model.target import TargetState
        return TargetState(tuple(p+v*t for p,v in zip(self.position,self.velocity)),self.velocity)


class ConstantSupport:
    def __call__(self,t,truth):
        return ''


def flight(library,missile,point,distance,target_speed):
    altitude,speed,delta,aspect=point
    horizontal=math.sqrt(max(0.,distance*distance-delta*delta))
    angle=math.radians(aspect)
    target=StraightTarget((horizontal,altitude+delta,0.),(-target_speed*math.cos(angle),0.,target_speed*math.sin(angle)))
    rt=library.create(library.profile(missile),launch_position_m=(0.,altitude,0.),
                      launch_velocity_mps=(speed,0.,0.),launch_heading_deg=0.,launch_pitch_deg=0.,
                      target=target,launcher_support=ConstantSupport(),**LAUNCH_OPTIONS)
    seeker=None
    seeker_range=library.info(missile).seeker_range_m
    while not rt.done:
        rt.step()
        state=target.state_at(rt.time_s)
        if seeker is None and math.dist(rt.state[:3],state.position)<=seeker_range:
            seeker=rt.time_s
        if len(rt._rows)>96:
            del rt._rows[:-1]
    return {'range_m':distance,'flight_s':rt.time_s,'seeker_on_s':seeker,'hit':rt.event=='fuse','event':rt.event}


def compute_cell(job):
    missile,point,options=job
    t0=time.perf_counter()
    library=default_library(options['missile_sim'])
    low=max(1000.,abs(point[2])+100.)
    high=options['max_range_m']
    at_low=flight(library,missile,point,low,options['target_speed_mps'])
    if not at_low['hit']:
        # A near-launch miss is not evidence that the entire range line is empty
        # (a large altitude difference can require room to turn).
        found=None
        for test in range(1,13):
            probe=low+(high-low)*test/12.
            result=flight(library,missile,point,probe,options['target_speed_mps'])
            if result['hit']:
                found=(probe,result)
                break
        if found is None:
            return dict(point=list(point),rmax_m=0.,samples=[],censored=False,wall_s=time.perf_counter()-t0)
        low,at_low=found
    at_high=flight(library,missile,point,high,options['target_speed_mps'])
    censored=at_high['hit']
    if censored:
        low,at_low=high,at_high
    else:
        while high-low>options['tolerance_m']:
            mid=(low+high)/2.
            result=flight(library,missile,point,mid,options['target_speed_mps'])
            if result['hit']:
                low,at_low=mid,result
            else:
                high=mid
    samples=[]
    for fraction in (.1,.25,.5,.75,1.):
        distance=max(abs(point[2])+100.,low*fraction)
        result=at_low if abs(distance-low)<1e-8 else flight(library,missile,point,distance,options['target_speed_mps'])
        if result['hit']:
            samples.append(result)
    samples=sorted({s['range_m']:s for s in samples}.values(),key=lambda s:s['range_m'])
    return dict(point=list(point),rmax_m=low,samples=samples,censored=censored,wall_s=time.perf_counter()-t0)


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=1))
    temporary.replace(path)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--missiles',default='us_aim_120c_5',help='comma separated; all uses active profiles with Pk models')
    ap.add_argument('--altitudes',default='2000,5000,8000,11000')
    ap.add_argument('--speeds',default='200,300,400')
    ap.add_argument('--delta-altitudes',default='-4000,0,4000')
    ap.add_argument('--aspects',default='0,90,180')
    ap.add_argument('--target-speed',type=float,default=300.)
    ap.add_argument('--max-range-km',type=float,default=120.)
    ap.add_argument('--tolerance-m',type=float,default=250.)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit',type=int,default=0,help='cap NEW cells across all missiles (0 = full grid)')
    ap.add_argument('--out',type=Path,default=ROOT/'data'/'reach')
    ap.add_argument('--missile-sim',default=None)
    args=ap.parse_args(argv)
    if not 1<=args.workers<=48 or args.tolerance_m<=0 or args.max_range_km<=0 or args.limit<0:
        ap.error('workers 1..48; positive range/tolerance; nonnegative limit')
    axes=dict(altitude_m=sorted(set(map(float,args.altitudes.split(',')))),
              speed_mps=sorted(set(map(float,args.speeds.split(',')))),
              delta_altitude_m=sorted(set(map(float,args.delta_altitudes.split(',')))),
              aspect_deg=sorted(set(map(float,args.aspects.split(',')))))
    if any(not all(math.isfinite(x) for x in axis) for axis in axes.values()):
        ap.error('grid values must be finite')
    options=dict(target_speed_mps=args.target_speed,max_range_m=args.max_range_km*1000.,
                 tolerance_m=args.tolerance_m,missile_sim=args.missile_sim)
    library=default_library(args.missile_sim)
    if args.missiles=='all':
        from wt_overlay import pk
        missiles=[mid for mid in pk.available() if library.is_active(library.info(mid).profile_id)]
    else:
        missiles=args.missiles.split(',')
    paths,data,todo={}, {}, []
    for missile in missiles:
        path=args.out/(missile+'.json')
        header=dict(version=1,missile=missile,axes=axes,options=options,
                    assumptions='D: straight target, unlimited tracked launcher support; reach is not Pk.',cells=[])
        if path.exists():
            previous=json.loads(path.read_text())
            if any(previous[k]!=header[k] for k in ('version','missile','axes','options')):
                raise ValueError('resume grid/options differ: '+str(path))
            header=previous
        paths[missile],data[missile]=path,header
        completed={tuple(c['point']) for c in header['cells']}
        for point in itertools.product(*axes.values()):
            if not 0<=point[0]+point[2]<=19500. or not 0<point[1] or not 0<=point[3]<=180.:
                continue
            if point not in completed:
                todo.append((missile,point,options))
    total=len(todo)
    chosen=todo[:args.limit] if args.limit else todo
    started=time.perf_counter()
    elapsed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(compute_cell,job):job[0] for job in chosen}
        for f in concurrent.futures.as_completed(futures):
            missile=futures[f]
            cell=f.result()
            data[missile]['cells'].append(cell)
            data[missile]['cells'].sort(key=lambda c:c['point'])
            save(paths[missile],data[missile])
            elapsed.append(cell['wall_s'])
            estimate=sum(elapsed)/len(elapsed)*total/args.workers
            print(json.dumps(dict(missile=missile,point=cell['point'],rmax_m=cell['rmax_m'],
                                  censored=cell['censored'],completed=len(elapsed),new_total=total,
                                  estimated_full_wall_s=estimate,elapsed_s=time.perf_counter()-started)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
