"""Reach-table lookup; no simulation in the training worker.

Tables contain straight, non-evading target results and timing samples. An
absent table leaves timing fields invalid and scripts use their existing Pk line.
Interpolation never fills missing cells with made-up timings.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

DATA_DIR=Path(__file__).resolve().parents[1]/'data'/'reach'


class ReachTable:
    def __init__(self, data):
        if data.get('version')!=1:
            raise ValueError('unsupported reach table version')
        self.data=data
        self.axes=data['axes']
        self.cells={tuple(c['point']):c for c in data['cells']}

    @classmethod
    def load(cls, missile, directory=None):
        path=Path(directory or DATA_DIR)/(missile+'.json')
        if not path.exists():
            return None
        return cls(json.loads(path.read_text()))

    def _weights(self, altitude,speed,delta,aspect):
        values=(altitude,speed,delta,aspect)
        names=('altitude_m','speed_mps','delta_altitude_m','aspect_deg')
        neighbours=[]
        for name,value in zip(names,values):
            axis=sorted(self.axes[name])
            if value<axis[0] or value>axis[-1]:
                return []
            lo=max(x for x in axis if x<=value)
            hi=min(x for x in axis if x>=value)
            if hi==lo:
                neighbours.append(((lo,1.),))
            else:
                f=(value-lo)/(hi-lo)
                neighbours.append(((lo,1-f),(hi,f)))
        out=[]
        for terms in itertools.product(*neighbours):
            point=tuple(t[0] for t in terms)
            cell=self.cells.get(point)
            if cell is None or cell.get('censored'):
                return []
            out.append((cell,math.prod(t[1] for t in terms)))
        return out

    def rmax(self, altitude,speed,aspect=0.,delta_altitude=0.):
        weights=self._weights(altitude,speed,delta_altitude,aspect)
        return None if not weights else sum(c['rmax_m']*w for c,w in weights)

    def times(self, altitude,speed,delta,aspect,distance):
        weights=self._weights(altitude,speed,delta,aspect)
        if not weights:
            return None
        result=[0.,0.]
        for cell,weight in weights:
            samples=cell['samples']
            if not samples or distance<samples[0]['range_m'] or distance>samples[-1]['range_m']:
                return None
            low=max((s for s in samples if s['range_m']<=distance),key=lambda s:s['range_m'])
            high=min((s for s in samples if s['range_m']>=distance),key=lambda s:s['range_m'])
            f=0. if high['range_m']==low['range_m'] else (distance-low['range_m'])/(high['range_m']-low['range_m'])
            for j,k in enumerate(('flight_s','seeker_on_s')):
                if low[k] is None or high[k] is None:
                    return None
                result[j]+=weight*(low[k]+f*(high[k]-low[k]))
        return tuple(result)
