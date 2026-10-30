/* Calibrated light wireframe style; no invented camera or resampled world state. */
(function(root){
const P=root.CalibratedProjection,edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const colors={vehicle:'#347ce5',pedestrian:'#e19a19',bicycle:'#9a58cc',cyclist:'#9a58cc'};
function rings(g,fn){if(g.type==='Polygon')fn(g.coordinates);else if(g.type==='MultiPolygon')g.coordinates.forEach(fn);else if(g.type==='GeometryCollection')g.geometries.forEach(x=>rings(x,fn));}
function segment(a,b,c){let p=P.cameraPoint(a,c),q=P.cameraPoint(b,c);if(p[2]<.15&&q[2]<.15)return null;if(p[2]<.15||q[2]<.15){const t=(.15-p[2])/(q[2]-p[2]),v=p.map((x,i)=>x+t*(q[i]-x));if(p[2]<.15)p=v;else q=v;}const out=[];for(let k=0;k<=12;k++){const t=k/12,v=P.project(p.map((x,i)=>x+t*(q[i]-x)),c);if(!v||Math.abs(v[0])>c.width*6||Math.abs(v[1])>c.height*6)return null;out.push(v);}return out;}
function draw(canvas,c,{features=[],layers=new Set(),selected=null,overlay=false}={}){
 const ctx=canvas.getContext('2d');if(!overlay){ctx.fillStyle='#f7f9fb';ctx.fillRect(0,0,canvas.width,canvas.height);
 for(const f of features){if(!layers.has(f.layer))continue;rings(f.geometry,rs=>{ctx.beginPath();let valid=false;for(const r of rs){const poly=P.polygon(r.map(p=>P.mapPoint(p,c)),c);if(poly.length<3)continue;valid=true;poly.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();}if(valid){ctx.fillStyle=f.layer==='walkways'?'#e3e6e9':f.layer==='crosswalks'?'#f0f3f5':'#acd0e8';ctx.fill('evenodd');}});}}
 for(const o of c.objects){if(o.hidden_owner)continue;ctx.strokeStyle=o.id===selected?'#007e92':!o.reconstructed?'#d75b9b':colors[o.kind]||'#df654c';ctx.lineWidth=o.id===selected?2.5:1.4;for(const [i,j]of edges){const ps=segment(o.corners[i],o.corners[j],c);if(!ps)continue;ctx.beginPath();ps.forEach((p,k)=>k?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();}}
}
root.WireWhitebox={draw};})(globalThis);