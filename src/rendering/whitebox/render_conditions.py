"""Rasterize calibrated semantic conditions using the existing projection engine."""
import json
from pathlib import Path

import numpy as np

from driversi.rendering.camera_review.exporter import corners
from driversi.rendering.camera_review.video_export import WhiteboxPainter, encode
from driversi.rendering.whitebox.conditioning import CAMERAS, write_json
from driversi.rendering.whitebox.exporter import sha256
from driversi.rendering.whitebox.projection_profiles import display_geometry, profile_description


def ribbons(features, width=.09):
    """Visual line proxies; retained semantics never claim observed road paint."""
    result=[]
    for feature in features:
        geo=feature['geometry']
        if geo['type']=='LineString': lines=[geo['coordinates']]
        elif geo['type']=='MultiLineString': lines=geo['coordinates']
        else:
            result.append(feature)
            continue
        if feature['layer'] != 'boundaries':
            continue
        for line in lines:
            from shapely.geometry import LineString
            line=list(LineString(line).simplify(.03,preserve_topology=True).coords)
            for a,b in zip(line,line[1:]):
                a,b=np.asarray(a[:2]),np.asarray(b[:2]);delta=b-a;n=np.linalg.norm(delta)
                if n<1e-6:continue
                normal=np.array([-delta[1],delta[0]])*width/n
                pts=np.array([a-normal,b-normal,b+normal,a+normal,a-normal]).tolist()
                result.append({**feature,'geometry':{'type':'Polygon','coordinates':[pts]}})
    return result


def camera_frame(bundle, frame, cam, geometry_profile=None):
    value=next(c.copy() for c in frame['cameras'] if c['camera_id']==cam)
    profile=geometry_profile or bundle.get('display_profile','legacy_mixed_v1')
    value, objects = display_geometry(frame, value, profile)
    value.update(objects=[dict(id=o['actor_id'], kind=o['kind'], instance_id=o['instance_id'],
                               semantic_color=o['semantic_color'],
                               corners=corners(o['pose_xyz_yaw'][:3],o['dimensions_lwh'],o['pose_xyz_yaw'][3]).tolist(),
                               hidden_owner=False,reconstructed=True) for o in objects],
                 map_to_reconstruction=np.eye(4).tolist(),map_offset=[0,0,0],map_ground_z=0,
                 semantic_conditioning=True)
    return value


def render(bundle_root, role='source', *, output=None, geometry_profile=None):
    root=Path(bundle_root);out=Path(output) if output else root/(role+'-renders')
    if out.exists():raise FileExistsError(out)
    bundle=json.loads((root/(role+'.json')).read_text());mp=json.loads((root/'map.json').read_text())
    if len(bundle['frames'])!=71:raise ValueError('71 complete samples required')
    profile = geometry_profile or bundle.get('display_profile','legacy_mixed_v1')
    projection = profile_description(profile, state_role=bundle["role"])
    data={'map_features':ribbons(mp['features']),'map_layers':mp['layers']}
    painter=WhiteboxPainter(data)
    painter.context.eval('''videoLayers.clear();
      for(const layer of ['generic_drivable_areas','road_segments','intersections','carpark_areas','lanes_polygons','walkways','crosswalks','boundaries']) videoLayers.add(layer);
      const layerOrder=['generic_drivable_areas','road_segments','intersections','carpark_areas','lanes_polygons','walkways','crosswalks','boundaries'];
      videoFeatures.sort((a,b)=>layerOrder.indexOf(a.layer)-layerOrder.indexOf(b.layer));''')
    # Clip screen geometry before CPU rasterization. Distorted polygons may have
    # kilometre-long off-screen edges; Pillow need not walk those pixels.
    painter.context.eval('''const originalConditionPolygon=CalibratedProjection.polygon;
    CalibratedProjection.polygon=function(points,c){
      let poly=originalConditionPolygon(points,c);
      for(const distance of [p=>p[0]+2,p=>c.width+2-p[0],p=>p[1]+2,p=>c.height+2-p[1]]){
        const out=[];for(let i=0;i<poly.length;i++){
          const a=poly[i],b=poly[(i+1)%poly.length],da=distance(a),db=distance(b);
          if(da>=0)out.push(a);if((da>=0)!=(db>=0)){const t=da/(da-db);out.push(a.map((v,j)=>v+t*(b[j]-v)));}
        }poly=out;
      }return poly;
    };''')
    # Separate color meaning from the legacy GS-asset-availability overlay.
    painter.context.eval('''const originalWhiteboxDraw=SemanticWhitebox.draw;
    SemanticWhitebox.draw=function(canvas,c,options){
      originalWhiteboxDraw(canvas,{...c,objects:[]},options);
      const P=CalibratedProjection,ctx=canvas.getContext('2d'),faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]], polys=[];
      for(const o of c.objects)for(const face of faces){let points=face.map(i=>o.corners[i]),p=P.polygon(points,c);if(p.length)polys.push({p,o,z:points.reduce((s,x)=>s+P.cameraPoint(x,c)[2],0)/4});}
      polys.sort((a,b)=>b.z-a.z);for(const {p,o} of polys){ctx.beginPath();p.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.closePath();ctx.fillStyle=o.semantic_color;ctx.fill();ctx.strokeStyle='#243441';ctx.lineWidth=1;ctx.stroke();}
    };''')
    out.mkdir(parents=True)
    receipts=[]
    for cam in CAMERAS:
        images={}
        for frame in bundle['frames']:
            image=painter.draw(camera_frame(bundle,frame,cam,profile));tick=frame['tick']
            file=out/f'{tick:04d}-{cam}.png';image.save(file)
            images[tick]=image
            receipts.append(dict(tick=tick,camera=cam,time_seconds=tick/10,file=file.name,sha256=sha256(file)))
        # [0,7): 70 video frames at10fps. Endpoint71 remains as PNG/JSON.
        info=encode(out/(cam+'.mp4'),images,list(range(70)),10)
        print(json.dumps({'camera':cam,**info}),flush=True)
    result=dict(kind='semantic_whitebox_video_condition',role=role,images=receipts,
                projection=projection,
                dimension_sampling=bundle.get('dimension_sampling', 'previous_annotation_hold_v1'),
                renderer_sha256=sha256(__file__),
                source_sha256=sha256(root/(role+'.json')),map_sha256=sha256(root/'map.json'),
                palette_sha256=sha256(root/'palette.json'),frame_count=71,video_frame_count=70,fps=10,
                video_interval='[0,7); endpoint at7s retained in PNG and JSON',
                renderer='existing calibrated distortion projection; depth-sorted cuboid-face approximation',
                limitations=['Occlusion is a cuboid proxy, not a dense depth-buffer ground truth.',
                             'Map is planar; centerlines do not assert lane paint.',
                             'Colors are semantic labels, never actual paint color.',
                             'Display boundary simplification tolerance 0.03m; original map geometry retained unchanged.',
                             'Navigation centerlines are retained in JSON but excluded from generation images.'],trainable=False)
    write_json(out/'manifest.json',result)
    return result
