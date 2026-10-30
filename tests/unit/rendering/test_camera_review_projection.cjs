const test=require('node:test'),assert=require('node:assert/strict');
const P=require('../../../src/rendering/camera_review/projection.js');
const c={camera_to_world:[[1,0,0,0],[0,1,0,0],[0,0,1,0]],K:[[100,0,50],[0,100,50],[0,0,1]],distortion:[0,0,0,0,0]};
test('Nerfstudio negative Z is optical forward, positive Y is image up',()=>{assert.deepEqual(P.cameraPoint([1,2,-10],c),[1,-2,10]);assert.deepEqual(P.project(P.cameraPoint([1,2,-10],c),c),[60,30]);});
test('behind-camera points are not projected',()=>{assert.equal(P.project(P.cameraPoint([1,2,10],c),c),null);});
test('distortion is actually applied',()=>{const d={...c,distortion:[.1,0,0,0,0]};assert.ok(Math.abs(P.project([1,0,1],d)[0]-160)<1e-10);});
test('effective translation used',()=>{const d={...c,camera_to_world:[[1,0,0,2],[0,1,0,3],[0,0,1,4]]};assert.deepEqual(P.project(P.cameraPoint([2,3,-6],d),d),[50,50]);});
test('near clipping preserves finite front intersection',()=>{const q=P.clip([[0,0,-1],[1,0,1],[0,1,1]]);assert.equal(q.length,4);assert.ok(q.every(p=>p[2]>=.149999999));});
test('behind-camera polygon discarded',()=>{assert.deepEqual(P.polygon([[0,0,1],[1,0,1],[0,1,1]],c),[]);});
test('map transform includes declared elevation and offset',()=>{const d={map_to_reconstruction:[[1,0,0,4],[0,1,0,5],[0,0,1,6],[0,0,0,1]],map_offset:[10,20,30],map_ground_z:2};assert.deepEqual(P.mapPoint([1,2],d),[15,27,38]);});

test('float camera uses matrix inverse, not transpose approximation',()=>{const d={...c,camera_to_world:[[1.000001,0,0,2],[0,1,0,3],[0,0,1,4]]};const pixel=P.project(P.cameraPoint([3.000001,3,-6],d),d);assert.ok(Math.abs(pixel[0]-60)<1e-10);});
test('road crossing camera plane remains visible after frustum clipping',()=>{const poly=P.polygon([[-50,-1,10],[50,-1,10],[50,-1,-80],[-50,-1,-80]],c);assert.ok(poly.length>0);assert.ok(poly.flat().every(Number.isFinite));});

test('raw NAVSIM negative radial polynomial cannot fold ground above horizon',()=>{const d={...c,distortion:[-.356123,.172545,0,0,-.05231]};const poly=P.polygon([[-50,-1,10],[50,-1,10],[50,-1,-80],[-50,-1,-80]],d);assert.ok(poly.length>0);assert.ok(poly.every(p=>p[1]>=50));});
