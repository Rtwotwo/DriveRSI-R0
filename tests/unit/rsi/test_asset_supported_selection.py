import numpy as np
import pytest
from driversi.rsi.decisions.asset_supported_selection import select


def test_asset_cannot_expand_cutoff_or_use_geographic_metadata():
    x=np.array([[1,0],[.8,.6],[0,1]],dtype=float)
    rows=[dict(token=str(i),source_log=str(i),source_group=str(i),history_continuous=True,
               heading=180,coordinates=[99999,99999]) for i in range(3)]
    assert select(x[0],x,rows,{"2"},top_k=2,limit=1)==[]
    actual=select(x[0],x,rows,{"1","2"},top_k=2,limit=1)
    assert actual[0]["token"]=="1" and actual[0]["raw_visual_rank"]==2
    rows[1].update(heading=0,coordinates=[0,0])
    assert select(x[0],x,rows,{"1","2"},top_k=2,limit=1)==actual
    rows[1]["history_continuous"]=False
    assert select(x[0],x,rows,{"1","2"},top_k=2,limit=1)==[]


def test_rank_is_global_before_source_deduplication():
    rows=[dict(token=str(i),source_log=str(i),source_group="a" if i<2 else "b",history_continuous=True) for i in range(3)]
    x=np.array([[1.,0],[1.,0],[.8,.6]])
    result=select(x[0],x,rows,{"1","2"},top_k=3,limit=2)
    assert [r["raw_visual_rank"] for r in result]==[2,3]
    with pytest.raises(ValueError,match="unit"):
        select(x[0],x*2,rows,{"2"},top_k=3)
