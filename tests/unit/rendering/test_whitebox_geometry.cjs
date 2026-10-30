const assert=require('node:assert/strict');
const G=require(process.argv[2] || '../../../src/rendering/whitebox/geometry.js');
const fs=[{time_seconds:0},{time_seconds:.5},{time_seconds:1}];
assert.equal(G.frameAt(fs,.49,.75).index,0);
assert.equal(G.frameAt(fs,.5,.75).index,1);
assert.equal(G.frameAt(fs,1.01,.75).available,false);
assert.equal(G.frameAt(fs,-.1,.75).available,false);
assert.equal(G.frameAt(fs,.3,.15).available,false);
assert.deepEqual(G.cameraPoint([10,0,2],{x:0,y:0,z:2,yaw:0,pitch:0}),[0,0,10]);
assert.deepEqual(G.project([0,0,10],100,80,50),[50,40]);
assert.equal(G.clipLine([0,0,-2],[0,0,-1]),null);
const cut=G.clipLine([0,0,-1],[2,0,1]);assert.ok(Math.abs(cut[0][2]-.3)<1e-12);assert.equal(cut[1][2],1);
const ring=G.clipRing([[-1,0,-1],[1,0,1],[1,1,1],[-1,1,-1]]);
assert.equal(ring.length,4);assert.ok(ring.every(p=>p[2]>=.3-1e-12));
const geometry={type:'Polygon',coordinates:[[[0,0],[5,0],[5,5],[0,0]],[[1,1],[2,1],[1,2],[1,1]]]};
assert.equal(G.primitives(geometry)[0].rings.length,2); // holes are not flattened away
const obj={pose_xyz_yaw:[10,20,100,Math.PI/2],dimensions_lwh:[4,2,3]};
const before=JSON.stringify(obj),box=G.corners(obj);
assert.equal(JSON.stringify(obj),before);assert.equal(box[0][2],0);assert.equal(box[4][2],3);
assert.ok(Math.abs(box[0][0]-9)<1e-9);assert.ok(Math.abs(box[0][1]-22)<1e-9);
console.log('whitebox geometry: timestamp, gaps, clipping, projection, holes and non-mutating planar display passed');

assert.equal(G.safeFar({x:0,y:0},[-500,-500,500,500],1000,500,650),200);
const limited=G.safeFar({x:490,y:0},[-500,-500,500,500],1000,500,650);
assert.ok(limited>0 && limited<10);
assert.equal(G.safeFar({x:510,y:0},[-500,-500,500,500],1000,500,650),0);
