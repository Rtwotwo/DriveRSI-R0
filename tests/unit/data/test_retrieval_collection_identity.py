import json
import sqlite3
from types import SimpleNamespace
import pytest
from driversi.data.catalog.retrieval_collection import active_collection_identity, KEY


def test_cache_namespace_tracks_registered_supplement_and_split():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
    c=SimpleNamespace(db=db,split='navtrain')
    assert active_collection_identity(c,'base') is None
    c._history_recovery_view=('base','recovery')
    with pytest.raises(ValueError,match='registration'):active_collection_identity(c,'base')
    v=dict(split='navtrain',base_profile='base',history_recovery_profile='recovery',supplement_identity='a')
    db.execute('INSERT INTO metadata VALUES (?,?)',(KEY,json.dumps(v)))
    a=active_collection_identity(c,'base')
    v['supplement_identity']='b'
    db.execute('UPDATE metadata SET value=?',(json.dumps(v),))
    assert a!=active_collection_identity(c,'base')
    c.split='navtest'
    with pytest.raises(ValueError,match='differ'):active_collection_identity(c,'base')
