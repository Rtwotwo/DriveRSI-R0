/* Effective Nerfstudio camera -> optical RDF -> calibrated OpenCV distorted pixels. */
(function(root){
  const mul=(m,p)=>m.map(r=>r.reduce((s,v,i)=>s+v*p[i],0));
  const inverses=new WeakMap();
  function cameraPoint(p,c){let inverse=inverses.get(c);if(!inverse){const [[a,b,z],[d,e,f],[g,h,i]]=c.camera_to_world;
    const det=a*(e*i-f*h)-b*(d*i-f*g)+z*(d*h-e*g);
    if(!Number.isFinite(det)||Math.abs(det)<1e-10)throw new Error('Invalid effective camera pose');
    inverse=[[e*i-f*h,z*h-b*i,b*f-z*e],[f*g-d*i,a*i-z*g,z*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]].map(row=>row.map(v=>v/det));inverses.set(c,inverse);}
    const q=p.map((v,i)=>v-c.camera_to_world[i][3]),t=mul(inverse,q);return [t[0],-t[1],-t[2]];}
  function project(p,c){if(p[2]<.15)return null;const x=p[0]/p[2],y=p[1]/p[2],r=x*x+y*y;
    const [k1,k2,p1,p2,k3]=c.distortion,s=1+k1*r+k2*r*r+k3*r*r*r;
    const a=x*s+2*p1*x*y+p2*(r+2*x*x),b=y*s+p1*(r+2*y*y)+2*p2*x*y;
    // Nerfstudio samples pixel centres at .5; canvas coordinates use those centres.
    const out=[c.K[0][0]*a+c.K[0][1]*b+c.K[0][2],c.K[1][1]*b+c.K[1][2]];
    return out.every(Number.isFinite)&&out.every(v=>Math.abs(v)<1e7)?out:null;}
  function clipPlane(poly,distance){let out=[];for(let i=0;i<poly.length;i++){const a=poly[i],b=poly[(i+1)%poly.length],da=distance(a),db=distance(b),ai=da>=0,bi=db>=0;if(ai)out.push(a);if(ai!==bi){const t=da/(da-db);out.push(a.map((v,j)=>v+t*(b[j]-v)));}}return out;}
  function clip(poly){return clipPlane(poly,p=>p[2]-.15);}
  // Polynomial distortion is not valid at arbitrary off-axis rays. Clip before
  // the first nonmonotonic radial derivative to avoid folded road polygons.
  const radii=new WeakMap();
  function radius(c){if(radii.has(c))return radii.get(c);const [k1,k2,,,k3]=c.distortion;let limit=2.8;
    for(let r=.02;r<=2.8;r+=.01){const u=r*r;if(1+3*k1*u+5*k2*u*u+7*k3*u*u*u<=0){limit=Math.max(.01,r-.02);break;}}
    radii.set(c,limit);return limit;}
  function frustum(poly,c){for(const plane of [p=>p[2]-.15,p=>300-p[2]])poly=clipPlane(poly,plane);
    const r=radius(c)*Math.cos(Math.PI/24);for(let i=0;i<24;i++){const a=i*Math.PI/12;poly=clipPlane(poly,p=>r*p[2]-Math.cos(a)*p[0]-Math.sin(a)*p[1]);}return poly;}
  function polygon(points,c){const raw=frustum(points.map(p=>cameraPoint(p,c)),c);if(raw.length<3)return [];
    let out=[];for(let i=0;i<raw.length;i++){const a=raw[i],b=raw[(i+1)%raw.length];for(let s=0;s<8;s++){
      const t=s/8,q=project(a.map((v,j)=>v+t*(b[j]-v)),c);if(!q)return [];out.push(q);}}return out;}
  function mapPoint(p,c){const q=[p[0]+c.map_offset[0],p[1]+c.map_offset[1],c.map_ground_z+c.map_offset[2]];
    return c.map_to_reconstruction.slice(0,3).map(row=>row[0]*q[0]+row[1]*q[1]+row[2]*q[2]+row[3]);}
  const api={mul,cameraPoint,project,clip,polygon,mapPoint};root.CalibratedProjection=api;
  if(typeof module!=="undefined")module.exports=api;
})(globalThis);
