"""Global epoch cursor shared by ranks; no model RNG or skipped real samples."""
import math
import random


class GlobalEpochCursor:
    def __init__(self, rows, seed, batch_size, world_size, synthetic_every=8):
        self.real=[i for i,r in enumerate(rows) if r['kind']=='original_navtrain']
        self.synthetic=[i for i,r in enumerate(rows) if r['kind']=='synthetic_navtrain']
        if not self.real or len(self.real)+len(self.synthetic)!=len(rows):
            raise ValueError('Official original navtrain and admitted synthetic rows required')
        if any(type(x) is not int or x<1 for x in (batch_size,world_size,synthetic_every)):
            raise ValueError('Positive batch/rank sizes and explicit mixing interval required')
        self.seed=seed;self.batch_size=batch_size;self.world_size=world_size
        self.global_batch=batch_size*world_size;self.synthetic_every=synthetic_every
        self.epoch=0;self.offset=0;self.draws=0;self.original_draws=0;self.synthetic_draws=0;self.padding_draws=0
        self._cached_epoch=None

    def _sequence(self):
        if self._cached_epoch==self.epoch:return self._indices,self._padding
        # Explicit local-adaptation recipe: retain original rows for provenance,
        # but draw only admitted synthetic rows. The original-only control uses
        # the same recipe with no synthetic rows and samples its original pool.
        if self.synthetic_every == 1:
            indices=list(self.synthetic or self.real)
            random.Random(self.seed+self.epoch).shuffle(indices)
            count=len(indices);padding=math.ceil(count/self.global_batch)*self.global_batch-count
            indices += [indices[i%count] for i in range(padding)]
            self._cached_epoch=self.epoch;self._indices=indices;self._padding=[False]*count+[True]*padding
            return indices,self._padding
        real=list(self.real);random.Random(self.seed+self.epoch).shuffle(real)
        indices=[];synthetic_cursor=self.epoch*(len(real)//(self.synthetic_every-1))
        for pos,index in enumerate(real):
            indices.append(index)
            if self.synthetic and (pos+1)%(self.synthetic_every-1)==0:
                indices.append(self.synthetic[synthetic_cursor%len(self.synthetic)]);synthetic_cursor+=1
        count=len(indices);padding=math.ceil(count/self.global_batch)*self.global_batch-count
        indices += [indices[i%count] for i in range(padding)]
        self._cached_epoch=self.epoch;self._indices=indices;self._padding=[False]*count+[True]*padding
        return indices,self._padding

    def next_batch(self):
        sequence,padding=self._sequence();start=self.offset;end=start+self.global_batch
        ids=sequence[start:end];flags=padding[start:end]
        if len(ids)!=self.global_batch:raise ValueError('Cursor not at a global batch boundary')
        before=dict(epoch=self.epoch,offset=self.offset)
        self.offset=end;self.draws+=len(ids)
        synth=set(self.synthetic)
        self.synthetic_draws+=sum(i in synth for i in ids)
        self.original_draws+=sum(i not in synth for i in ids)
        self.padding_draws+=sum(flags)
        if end==len(sequence):self.epoch+=1;self.offset=0
        return ids,dict(before=before,padding=flags)

    def state_dict(self):
        return {k:getattr(self,k) for k in ('seed','batch_size','world_size','synthetic_every',
            'epoch','offset','draws','original_draws','synthetic_draws','padding_draws')}

    def load_state_dict(self,state):
        expected=self.state_dict()
        if set(state)!=set(expected):raise ValueError('Malformed epoch cursor')
        for k in ('seed','batch_size','world_size','synthetic_every'):
            if state[k]!=expected[k]:raise ValueError('Cursor recipe or world size changed')
        if any(type(v) is not int or v<0 for v in state.values()):raise ValueError('Invalid epoch cursor')
        for k,v in state.items():setattr(self,k,v)
        self._cached_epoch=None
        if self.offset>=len(self._sequence()[0]) or self.offset%self.global_batch:
            raise ValueError('Cursor is not a valid next global batch')
        if self.draws!=self.original_draws+self.synthetic_draws or self.padding_draws>self.draws:
            raise ValueError('Cursor exposure accounting differs')


def distributed_settings(settings):
    value=settings.get('distributed_training')
    keys={'devices','checkpoint_interval','synthetic_every'}
    if not isinstance(value,dict) or set(value)!=keys:
        raise ValueError('Explicit distributed devices/checkpoint interval/mixing required')
    devices=value['devices']
    if not isinstance(devices,list) or not 1<=len(devices)<=8 or len(devices)!=len(set(devices)) or any(type(x)is not int or x<0 for x in devices):
        raise ValueError('Explicit distinct local GPU indices required')
    if type(value['checkpoint_interval']) is not int or value['checkpoint_interval']<1 or type(value['synthetic_every']) is not int or value['synthetic_every']<1:
        raise ValueError('Positive checkpoint interval and sampling interval required;1 selects synthetic-only')
    return dict(value)


def remaining_updates(schedule_total, completed, attempt_cap):
    if any(type(v) is not int for v in (schedule_total, completed, attempt_cap)) or schedule_total < 1 or completed < 0 or attempt_cap < 1:
        raise ValueError('Explicit nonnegative schedule position and positive cap required')
    if completed >= schedule_total:
        raise ValueError('Native schedule already completed')
    return min(attempt_cap, schedule_total - completed)
