/* Pure display geometry. Recorded coordinates are never modified. */
(function(root){
  const G = {};
  G.frameAt = function(frames,t,maxAge){
    let lo=0, hi=frames.length;
    while(lo<hi){const m=(lo+hi)>>1;if(frames[m].time_seconds<=t+1e-8)lo=m+1;else hi=m;}
    const index=lo-1, frame=frames[Math.max(0,index)], age=t-frame.time_seconds;
    const ended=t>frames[frames.length-1].time_seconds+1e-6;
    return {index,frame,age,ended,available:index>=0&&!ended&&age<=maxAge+1e-8};
  };
  G.cameraPoint = function(p,c){
    const x=p[0]-c.x,y=p[1]-c.y,z=p[2]-c.z,s=Math.sin(c.yaw),a=Math.cos(c.yaw),sp=Math.sin(c.pitch),cp=Math.cos(c.pitch);
    return [x*s-y*a, (x*a+y*s)*sp+z*cp, (x*a+y*s)*cp-z*sp];
  };
  G.safeFar = function(c,bounds,w,h,f){
    const radius=Math.max(0,Math.min(c.x-bounds[0],c.y-bounds[1],bounds[2]-c.x,bounds[3]-c.y));
    return Math.min(200,radius/Math.hypot(1,w/(2*f),h/(2*f)));
  };
  G.project = (p,w,h,f) => [w/2+f*p[0]/p[2],h/2-f*p[1]/p[2]];
  const mix=(a,b,t)=>a.map((x,i)=>x+t*(b[i]-x));
  G.clipLine = function(a,b,near=.3,far=200){
    let u=0,v=1,d=b[2]-a[2];
    if(Math.abs(d)<1e-12)return a[2]>=near&&a[2]<=far?[a,b]:null;
    let x=(near-a[2])/d,y=(far-a[2])/d;if(x>y)[x,y]=[y,x];
    u=Math.max(u,x);v=Math.min(v,y);return u<=v?[mix(a,b,u),mix(a,b,v)]:null;
  };
  G.clipRing = function(points,near=.3,far=200){
    let out=points;
    for(const [z,greater] of [[near,true],[far,false]]){
      const input=out;out=[];if(!input.length)break;
      let a=input[input.length-1],ain=greater?a[2]>=z:a[2]<=z;
      for(const b of input){const bin=greater?b[2]>=z:b[2]<=z;
        if(ain!==bin)out.push(mix(a,b,(z-a[2])/(b[2]-a[2])));
        if(bin)out.push(b);a=b;ain=bin;
      }
    }
    return out;
  };
  G.corners = function(obj){
    const [x,y,,yaw]=obj.pose_xyz_yaw,[l,w,h]=obj.dimensions_lwh,c=Math.cos(yaw),s=Math.sin(yaw);
    return [0,h].flatMap(z=>[[1,1],[1,-1],[-1,-1],[-1,1]].map(([u,v])=>[x+c*u*l/2-s*v*w/2,y+s*u*l/2+c*v*w/2,z]));
  };
  G.edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  G.primitives=function(geometry){
    if(geometry.type==='Polygon')return [{type:'polygon',rings:geometry.coordinates}];
    if(geometry.type==='MultiPolygon')return geometry.coordinates.map(rings=>({type:'polygon',rings}));
    if(geometry.type==='LineString')return [{type:'line',points:geometry.coordinates}];
    if(geometry.type==='MultiLineString')return geometry.coordinates.map(points=>({type:'line',points}));
    if(geometry.type==='GeometryCollection')return geometry.geometries.flatMap(G.primitives);
    return [];
  };
  if(typeof module!=='undefined')module.exports=G;
  root.WhiteboxGeometry=G;
})(typeof window==='undefined'?globalThis:window);
