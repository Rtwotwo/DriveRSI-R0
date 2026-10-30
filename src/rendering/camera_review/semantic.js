/* Shared calibrated annotation drawing for the viewer and video export. */
(function(root){
  const P=root.CalibratedProjection;
const faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]];
const colors={lanes_polygons:'#374355',generic_drivable_areas:'#293440',intersections:'#303d4b',
 walkways:'#295346',crosswalks:'#6b5784',stop_polygons:'#745b3b',traffic_lights:'#edbd59',
 traffic_signs:'#edbd59',boundaries:'#738190',baseline_paths:'#536578',carpark_areas:'#334344'};
function geometry(g,fn){if(g.type==='Polygon')g.coordinates.forEach(x=>fn(x,true));else if(g.type==='MultiPolygon')g.coordinates.forEach(x=>x.forEach(y=>fn(y,true)));else if(g.type==='LineString')fn(g.coordinates,false);else if(g.type==='MultiLineString')g.coordinates.forEach(x=>fn(x,false));else if(g.type==='Point')fn([g.coordinates],false);else if(g.type==='MultiPoint')g.coordinates.forEach(x=>fn([x],false));else if(g.type==='GeometryCollection')g.geometries.forEach(x=>geometry(x,fn));}
function path(ctx,points){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));}
function fill(ctx,points,color,stroke){if(points.length<3)return;path(ctx,points);ctx.closePath();ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}

function draw(canvas,cam,options={}){const {overlay=false,features=[],layers=new Set(),selected=null,onlyMissing=false}=options;const ctx=canvas.getContext('2d');if(!overlay){ctx.fillStyle='#dbe4ee';ctx.fillRect(0,0,canvas.width,canvas.height);
 for(const feature of features){if(!layers.has(feature.layer))continue;geometry(feature.geometry,(points,closed)=>{if(!closed)return;const poly=P.polygon(points.map(p=>P.mapPoint(p,cam)),cam);fill(ctx,poly,colors[feature.layer]||'#bdc9d7');});}}
 const polygons=[];for(const obj of cam.objects){if(obj.hidden_owner)continue;const missing=!obj.reconstructed;
  if(!cam.source_frame_token&&onlyMissing&&!missing&&selected!==obj.id)continue;
  for(const face of faces){const world=face.map(i=>obj.corners[i]),poly=P.polygon(world,cam);if(!poly.length)continue;
   const depth=world.reduce((s,p)=>s+P.cameraPoint(p,cam)[2],0)/4;
   polygons.push({poly,depth,obj,missing});}}
 polygons.sort((a,b)=>b.depth-a.depth);for(const v of polygons){const chosen=v.obj.id===selected;
  const stroke=chosen?'#13bfa4':v.missing?'#d75b9b':'#657a8c';
  fill(ctx,v.poly,overlay?'rgba(0,0,0,0)':chosen?'#9ce6d5':v.missing?'#f6d2e4':'#f5f7f9',stroke);}
}
  const api={draw};root.SemanticWhitebox=api;
  if(typeof module!=="undefined")module.exports=api;
})(globalThis);
