/* Retained display geometry. All source records remain separate from the world.
 * Virtual inspection cameras are NOT sensor cameras and never relabel RGB.
 */
(function(root){
const G=root.WhiteboxGeometry;
const surfaces={generic_drivable_areas:['#d6e3ec',0],road_segments:['#c8dce9',1],lane_groups_polygons:['#c3dbe9',2],lane_group_connectors:['#c2d8e6',2],intersections:['#bdd7e7',3],lanes_polygons:['#acd0e8',4],gen_lane_connectors_scaled_width_polygons:['#b0d1e7',4],carpark_areas:['#d6e1e6',3],walkways:['#e3e6e9',5],crosswalks:['#fafbfc',6],stop_polygons:['#e4c9a4',7]};
const kindColors={vehicle:'#347ce5',pedestrian:'#df9117',bicycle:'#985cc6',cyclist:'#985cc6',generic_object:'#d9673b',traffic_cone:'#de693b',barrier:'#b85d30'};
function primitives(g){switch(g.type){case'Point':return[{type:'point',points:[g.coordinates]}];case'MultiPoint':return g.coordinates.map(p=>({type:'point',points:[p]}));case'GeometryCollection':return g.geometries.flatMap(primitives);default:return G.primitives(g);}}
function bounds(p){const pts=p.rings?p.rings.flat():p.points;return [Math.min(...pts.map(p=>p[0])),Math.min(...pts.map(p=>p[1])),Math.max(...pts.map(p=>p[0])),Math.max(...pts.map(p=>p[1]))];}
function countKinds(objects){const c={};for(const o of objects)c[o.kind]=(c[o.kind]||0)+1;return c;}
function frameObjects(scene,frame,source){const kinds=new Map(scene.source_frames.flatMap(f=>f.objects.map(o=>[o.id,o.kind])));return{
 source:source.objects.map(o=>({...o,role:'source',pose_xyz_yaw:o.pose,dimensions_lwh:o.dimensions})),
 world:frame.actors.filter(a=>a.present).map(a=>({...a,kind:kinds.get(a.id)||'generic_object',pose_xyz_yaw:[...a.position,0,a.yaw],dimensions_lwh:a.dimensions,role:a.controlled?'controlled':'replay'}))
};}
function camera(anchor,mode,orbit){const [x,y,,yaw]=anchor.pose_xyz_yaw;
 if(mode==='front')return{x:x+anchor.dimensions_lwh[0]*.27*Math.cos(yaw),y:y+anchor.dimensions_lwh[0]*.27*Math.sin(yaw),z:1.65,yaw,pitch:.04};
 if(mode==='chase')return{x:x-18*Math.cos(yaw),y:y-18*Math.sin(yaw),z:10,yaw,pitch:.4};
 if(mode==='bird')return{x:x-.1,y:y-.1,z:65,yaw,pitch:1.565};
 const a=yaw+orbit.yaw,d=orbit.distance,cp=Math.cos(orbit.pitch);return{x:x+orbit.x-d*cp*Math.cos(a),y:y+orbit.y-d*cp*Math.sin(a),z:1+d*Math.sin(orbit.pitch),yaw:a,pitch:orbit.pitch};
}
function build(features){const items=[];const stats={};
 for(const f of features){stats[f.layer]=(stats[f.layer]||0)+1;for(const p of primitives(f.geometry))items.push({...p,bounds:bounds(p),layer:f.layer,id:f.id,properties:f.properties||{}});}
 items.sort((a,b)=>(surfaces[a.layer]?.[1]??8)-(surfaces[b.layer]?.[1]??8));
 return{items,stats,count:features.length};
}
function render(canvas,prepared,options){const {camera:c,objects,sourceSignals,layers,selected,kinds,ghosts=[],trajectory=[],showTrajectory=true,showSignals=true,showLabels=true,hideActor=null}=options,ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,f=w*.65,far=220;
 ctx.fillStyle='#f7f9fb';ctx.fillRect(0,0,w,h);ctx.lineWidth=1;const hits=[],visible=new Set(),mapVisible=new Set(),signals=new Map((sourceSignals?.records||[]).map(s=>[String(s.connector_id),s.is_red]));
 function line(a,b,color,width=1,dash=[]){const ps=G.clipLine(G.cameraPoint(a,c),G.cameraPoint(b,c),.3,far);if(!ps)return;const p=G.project(ps[0],w,h,f),q=G.project(ps[1],w,h,f);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);}
 for(const it of prepared.items){if(!layers.has(it.layer))continue;const [x,y,X,Y]=it.bounds,dx=Math.max(x-c.x,0,c.x-X),dy=Math.max(y-c.y,0,c.y-Y);if(dx*dx+dy*dy>far*far)continue;
  if(it.type==='polygon'){ctx.beginPath();let drawn=false;for(const ring of it.rings){const clip=G.clipRing(ring.map(p=>G.cameraPoint([p[0],p[1],0],c)),.3,far);if(clip.length<3)continue;drawn=true;clip.forEach((p,i)=>{const q=G.project(p,w,h,f);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();}
   if(drawn){ctx.fillStyle=surfaces[it.layer]?.[0]||'#c9d9e3';ctx.fill('evenodd');mapVisible.add(it.id);}
  }
 }
 for(const it of prepared.items){if(!layers.has(it.layer))continue;const [x,y,X,Y]=it.bounds;if(Math.max(x-c.x,0,c.x-X)**2+Math.max(y-c.y,0,c.y-Y)**2>far*far)continue;
  if(it.type==='line'){const id=it.id.split(':').at(-1),red=it.layer==='lane_connectors'&&showSignals?signals.get(id):undefined;
   const color=red===true?'#e44648':red===false?'#bd840c':it.layer==='boundaries'?'#6d8e9f':it.layer==='lane_connectors'?'#498b9c':'#a88e43';
   for(let i=1;i<it.points.length;i++)line([...it.points[i-1].slice(0,2),.025],[...it.points[i].slice(0,2),.025],color,red!==undefined?2.8:1.2,it.layer==='baseline_paths'?[5,5]:[]);mapVisible.add(it.id);
  }else if(it.type==='point'){const p=it.points[0],v=G.cameraPoint([p[0],p[1],.1],c);if(v[2]<.3||v[2]>far)continue;const q=G.project(v,w,h,f);if(q[0]<0||q[0]>w||q[1]<0||q[1]>h)continue;
   ctx.strokeStyle=it.layer==='traffic_lights'?'#d89e22':'#668497';ctx.lineWidth=1.5;ctx.beginPath();
   if(it.layer==='traffic_lights'){ctx.rect(q[0]-4,q[1]-10,8,13);ctx.moveTo(q[0],q[1]+3);ctx.lineTo(q[0],q[1]+9);}
   else{ctx.moveTo(q[0]-3,q[1]);ctx.lineTo(q[0]+3,q[1]);ctx.moveTo(q[0],q[1]-3);ctx.lineTo(q[0],q[1]+3);}
   ctx.stroke();mapVisible.add(it.id);
  }
 }
 function object(o,ghost){if(!kinds.has(o.kind)||o.id===hideActor)return;const chosen=o.id===selected,ps=G.corners(o),color=ghost?'#ad72b6':chosen?'#007e92':o.collision?'#cf303f':o.reconstructed===false?'#d66ca1':kindColors[o.kind]||'#d9673b';for(const [a,b]of G.edges)line(ps[a],ps[b],color,chosen?2.7:1.7,ghost?[5,4]:[]);
  const center=G.cameraPoint([o.pose_xyz_yaw[0],o.pose_xyz_yaw[1],o.dimensions_lwh[2]+.35],c);if(center[2]>.3&&center[2]<far){const q=G.project(center,w,h,f);if(q[0]>=0&&q[0]<=w&&q[1]>=0&&q[1]<=h){visible.add(o.id);if(showLabels||chosen){ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText((o.id==='ego'?'ego':o.id.slice(0,6))+(ghost?' · 仅日志':''),q[0]+3,q[1]);}hits.push({x:q[0],y:q[1],id:o.id});}}
 }
 ghosts.forEach(o=>object(o,true));objects.forEach(o=>object(o,false));
 if(showTrajectory){let last=null;for(const p of trajectory){if(p&&last)line([...last.slice(0,2),.09],[...p.slice(0,2),.09],'#00a68f',3);last=p;}}
 return{hits,visibleObjects:visible.size,totalObjects:objects.length,filteredObjects:objects.filter(o=>kinds.has(o.kind)).length,nearbyMapFeatures:mapVisible.size,loadedMapFeatures:prepared.count};
}
root.StructuralWhitebox={build,render,frameObjects,camera,primitives,countKinds};
if(typeof module!=='undefined')module.exports=root.StructuralWhitebox;
})(globalThis);